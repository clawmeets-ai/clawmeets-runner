# SPDX-License-Identifier: MIT
"""
clawmeets/runner/home_fs_win.py

Windows primitives for the home-folder path guard (``home_fs.HomeFs``).

``WindowsOps`` mirrors ``home_fs._PosixOps`` one method for one method, so
the guard's logic (tiers, identity map, device check, two-pass delete) is
shared and only the way a single name is opened differs:

* Every open is ``NtCreateFile`` with ``RootDirectory`` set to the parent's
  handle — the Windows equivalent of ``openat(dir_fd, ...)``; Win32 path
  parsing (drive letters, ``\\\\?\\``, trailing-dot stripping, device names)
  never runs on a name taken from a request.
* Every open passes ``FILE_OPEN_REPARSE_POINT``, so a symlink, junction,
  mount point or any other reparse point is opened as itself and never
  followed. A link-type reparse point (symlink, junction, mount point) is
  ``kind="symlink"``: listed, delete-only. Any other reparse point (OneDrive
  placeholder, dedup, compressed file) is the data itself, so it is
  ``kind="other"`` with ``reparse=True`` and is never deleted — deleting a
  placeholder would delete the cloud copy too. A folder that turns out to be
  either is refused on its own handle, which closes the swap-mid-operation
  race exactly as ``O_NOFOLLOW`` does on POSIX.
* After every open of an existing name, the object's real name is read back
  from the handle; if it does not fold to the requested name, the open
  went through an alias (an 8.3 short name such as ``CREDEN~1.JSO``) and is
  refused.
* Identity is (volume serial number, 128-bit file id), so the protected-
  identity map and the other-volume check work as on POSIX.
* Deletes set the delete disposition on a handle opened by name without
  following, with POSIX semantics where the filesystem supports them.

File handles are wrapped in CRT descriptors (``msvcrt.open_osfhandle``) so
``os.read`` / ``os.write`` / ``os.fsync`` / ``os.close`` in the shared code
work unchanged; folder handles stay raw ``HANDLE`` values.
"""
from __future__ import annotations

import ctypes
import errno
import msvcrt
import os
from ctypes import wintypes
from pathlib import Path

from clawmeets.runner.home_fs import HomeFsError, _fold, _Stat, _unsupported_entry

_ntdll = ctypes.WinDLL("ntdll")
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Access rights
_FILE_READ_DATA = 0x0001          # = FILE_LIST_DIRECTORY on a folder
_FILE_TRAVERSE = 0x0020
_FILE_READ_ATTRIBUTES = 0x0080
_DELETE = 0x00010000
_SYNCHRONIZE = 0x00100000
_GENERIC_WRITE = 0x40000000
# NtCreateFile dispositions and options
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_SHARE_ALL = 0x7
_OBJ_CASE_INSENSITIVE = 0x40
# File attributes
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_NORMAL = 0x80
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
# GetFinalPathNameByHandleW flags
_FILE_NAME_NORMALIZED = 0x0
_VOLUME_NAME_NONE = 0x4
# Information classes
_FileFullDirectoryInfo = 14
_FileFullDirectoryRestartInfo = 15
_FileIdInfo = 18
_FileAttributeTagInfo = 9
# IsReparseTagNameSurrogate: the tag names another file (symlink, junction /
# mount point, WSL symlink). Every other tag (cloud placeholder, dedup, WOF
# compression, AppExecLink, AF_UNIX) is the data itself.
_REPARSE_TAG_NAME_SURROGATE = 0x20000000
_FileRenameInformation = 10       # NtSetInformationFile
_FileDispositionInformation = 13
_FileDispositionInformationEx = 64
_FILE_DISPOSITION_DELETE = 0x1
_FILE_DISPOSITION_POSIX_SEMANTICS = 0x2
_FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE = 0x10
_ERROR_NO_MORE_FILES = 18
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_INVALID_HANDLE_VALUES = (None, -1, ctypes.c_void_p(-1).value)
_EPOCH_AS_FILETIME = 116444736000000000

# NTSTATUS -> errno, for the codes the guard tells apart. Anything else is
# translated through RtlNtStatusToDosError.
_STATUS_ERRNO = {
    0xC000000F: errno.ENOENT,     # STATUS_NO_SUCH_FILE
    0xC0000034: errno.ENOENT,     # STATUS_OBJECT_NAME_NOT_FOUND
    0xC000003A: errno.ENOENT,     # STATUS_OBJECT_PATH_NOT_FOUND
    0xC0000056: errno.ENOENT,     # STATUS_DELETE_PENDING
    0xC0000035: errno.EEXIST,     # STATUS_OBJECT_NAME_COLLISION
    0xC0000103: errno.ENOTDIR,    # STATUS_NOT_A_DIRECTORY
    0xC00000BA: errno.EISDIR,     # STATUS_FILE_IS_A_DIRECTORY
    0xC0000101: errno.ENOTEMPTY,  # STATUS_DIRECTORY_NOT_EMPTY
    0xC0000022: errno.EACCES,     # STATUS_ACCESS_DENIED
    0xC0000043: errno.EACCES,     # STATUS_SHARING_VIOLATION
    0xC0000121: errno.EACCES,     # STATUS_CANNOT_DELETE
    0xC0000033: errno.EINVAL,     # STATUS_OBJECT_NAME_INVALID
}
# Codes meaning "this filesystem has no FileDispositionInformationEx".
_STATUS_NO_EX = {0xC0000003, 0xC000000D, 0xC00000BB}


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p)]


class _OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(_UNICODE_STRING)), ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", ctypes.c_void_p), ("SecurityQualityOfService", ctypes.c_void_p)]


class _IO_STATUS_BLOCK(ctypes.Structure):
    _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [("dwFileAttributes", wintypes.DWORD), ("ftCreationTime", wintypes.FILETIME),
                ("ftLastAccessTime", wintypes.FILETIME), ("ftLastWriteTime", wintypes.FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD), ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD), ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD), ("nFileIndexLow", wintypes.DWORD)]


class _FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]


class _FILE_ID_INFO(ctypes.Structure):
    _fields_ = [("VolumeSerialNumber", ctypes.c_ulonglong), ("FileId", ctypes.c_ubyte * 16)]


class _FILE_RENAME_INFORMATION(ctypes.Structure):
    _fields_ = [("ReplaceIfExists", wintypes.BOOLEAN), ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.ULONG), ("FileName", wintypes.WCHAR * 1)]


_NtCreateFile = _ntdll.NtCreateFile
_NtCreateFile.restype = ctypes.c_long
_NtCreateFile.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.ULONG, ctypes.POINTER(_OBJECT_ATTRIBUTES),
                          ctypes.POINTER(_IO_STATUS_BLOCK), ctypes.c_void_p, wintypes.ULONG, wintypes.ULONG,
                          wintypes.ULONG, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
_NtSetInformationFile = _ntdll.NtSetInformationFile
_NtSetInformationFile.restype = ctypes.c_long
_NtSetInformationFile.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IO_STATUS_BLOCK), ctypes.c_void_p,
                                  wintypes.ULONG, ctypes.c_int]
_RtlNtStatusToDosError = _ntdll.RtlNtStatusToDosError
_RtlNtStatusToDosError.restype = wintypes.ULONG
_RtlNtStatusToDosError.argtypes = [wintypes.ULONG]

_CreateFileW = _kernel32.CreateFileW
_CreateFileW.restype = wintypes.HANDLE
_CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
_CloseHandle = _kernel32.CloseHandle
_CloseHandle.restype = wintypes.BOOL
_CloseHandle.argtypes = [wintypes.HANDLE]
_GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
_GetFileInformationByHandle.restype = wintypes.BOOL
_GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)]
_GetFileInformationByHandleEx = _kernel32.GetFileInformationByHandleEx
_GetFileInformationByHandleEx.restype = wintypes.BOOL
_GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
_GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
_GetFinalPathNameByHandleW.restype = wintypes.DWORD
_GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]


def _nt_error(status: int, name: str) -> OSError:
    code = status & 0xFFFFFFFF
    mapped = _STATUS_ERRNO.get(code)
    if mapped is not None:
        return OSError(mapped, f"{os.strerror(mapped)} (NTSTATUS 0x{code:08X})", name)
    winerr = _RtlNtStatusToDosError(code)
    return OSError(0, ctypes.FormatError(winerr), name, winerr)


def _last_error() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())


def _close(h: int) -> None:
    _CloseHandle(h)


def _nt_open(parent: int, name: str, access: int, disposition: int, options: int) -> int:
    """Open ``name`` relative to the folder handle ``parent`` without
    following a reparse point in it."""
    buf = ctypes.create_unicode_buffer(name)
    nbytes = (len(buf) - 1) * ctypes.sizeof(wintypes.WCHAR)
    if nbytes > 0xFFFC:
        raise OSError(errno.ENAMETOOLONG, "name is too long", name)
    us = _UNICODE_STRING(nbytes, nbytes + ctypes.sizeof(wintypes.WCHAR), ctypes.addressof(buf))
    oa = _OBJECT_ATTRIBUTES(ctypes.sizeof(_OBJECT_ATTRIBUTES), parent, ctypes.pointer(us),
                            _OBJ_CASE_INSENSITIVE, None, None)
    h = wintypes.HANDLE()
    iosb = _IO_STATUS_BLOCK()
    status = _NtCreateFile(
        ctypes.byref(h), access | _SYNCHRONIZE, ctypes.byref(oa), ctypes.byref(iosb), None,
        _FILE_ATTRIBUTE_NORMAL, _FILE_SHARE_ALL, disposition,
        options | _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT, None, 0,
    )
    if status < 0:
        raise _nt_error(status, name)
    return h.value


def _real_name(h: int) -> str:
    """Last component of the object's real (long) name, from its handle.

    ``FILE_NAME_NORMALIZED`` expands 8.3 short names; the plain name query
    (``FileNameInfo``) returns the name as it was opened and would not.
    """
    buf = ctypes.create_unicode_buffer(32768)
    n = _GetFinalPathNameByHandleW(h, buf, len(buf), _FILE_NAME_NORMALIZED | _VOLUME_NAME_NONE)
    if not n or n >= len(buf):
        raise _last_error()
    return buf.value.rsplit("\\", 1)[-1]


def _check_alias(h: int, requested: str) -> None:
    """Refuse an open that resolved through a different name — an 8.3
    short name (``PROGRA~1``) or any other alias the name checks cannot
    see. Case differences are fine: identity covers them."""
    if _fold(_real_name(h)) != _fold(requested):
        raise HomeFsError("short_name", "short (8.3) names and other aliases are not accepted; use the full name")


def _stat_handle(h: int) -> _Stat:
    info = _BY_HANDLE_FILE_INFORMATION()
    if not _GetFileInformationByHandle(h, ctypes.byref(info)):
        raise _last_error()
    attrs = info.dwFileAttributes
    reparse = False
    if attrs & _FILE_ATTRIBUTE_REPARSE_POINT:
        tag = _FILE_ATTRIBUTE_TAG_INFO()
        if not _GetFileInformationByHandleEx(h, _FileAttributeTagInfo, ctypes.byref(tag), ctypes.sizeof(tag)):
            raise _last_error()
        if tag.ReparseTag & _REPARSE_TAG_NAME_SURROGATE:
            kind = "symlink"
        else:
            kind, reparse = "other", True
    elif attrs & _FILE_ATTRIBUTE_DIRECTORY:
        kind = "dir"
    else:
        kind = "file"
    ino = (info.nFileIndexHigh << 32) | info.nFileIndexLow
    fid = _FILE_ID_INFO()
    if _GetFileInformationByHandleEx(h, _FileIdInfo, ctypes.byref(fid), ctypes.sizeof(fid)):
        ino = int.from_bytes(bytes(fid.FileId), "little") or ino
    ft = info.ftLastWriteTime
    mtime = (((ft.dwHighDateTime << 32) | ft.dwLowDateTime) - _EPOCH_AS_FILETIME) / 1e7
    return _Stat(
        kind=kind, nlink=info.nNumberOfLinks, dev=info.dwVolumeSerialNumber, ino=ino,
        size=(info.nFileSizeHigh << 32) | info.nFileSizeLow, mtime=mtime, reparse=reparse,
    )


def _set_delete(h: int, name: str) -> None:
    iosb = _IO_STATUS_BLOCK()
    flags = wintypes.ULONG(_FILE_DISPOSITION_DELETE | _FILE_DISPOSITION_POSIX_SEMANTICS
                           | _FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE)
    status = _NtSetInformationFile(h, ctypes.byref(iosb), ctypes.byref(flags), ctypes.sizeof(flags),
                                   _FileDispositionInformationEx)
    if status < 0 and (status & 0xFFFFFFFF) in _STATUS_NO_EX:
        delete = wintypes.BOOLEAN(True)
        status = _NtSetInformationFile(h, ctypes.byref(iosb), ctypes.byref(delete), ctypes.sizeof(delete),
                                       _FileDispositionInformation)
    if status < 0:
        raise _nt_error(status, name)


class WindowsOps:
    """``home_fs._PosixOps`` for Windows. See the module docstring."""

    def open_root(self, path: Path) -> int:
        h = _CreateFileW(
            str(path), _FILE_READ_DATA | _FILE_TRAVERSE | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
            _FILE_SHARE_ALL, None, _OPEN_EXISTING, _FILE_FLAG_BACKUP_SEMANTICS, None,
        )
        if h in _INVALID_HANDLE_VALUES:
            raise _last_error()
        return h

    def open_dir(self, parent: int, name: str) -> int:
        h = _nt_open(parent, name, _FILE_READ_DATA | _FILE_TRAVERSE | _FILE_READ_ATTRIBUTES,
                     _FILE_OPEN, _FILE_DIRECTORY_FILE)
        try:
            _check_alias(h, name)
            st = _stat_handle(h)
            if st.reparse:
                raise _unsupported_entry()
            if st.kind != "dir":
                raise HomeFsError("symlink", "symlinks and junctions are never followed")
        except BaseException:
            _close(h)
            raise
        return h

    def close_dir(self, h: int) -> None:
        _close(h)

    def stat_dir(self, h: int) -> _Stat:
        return _stat_handle(h)

    def stat_file(self, fd: int) -> _Stat:
        return _stat_handle(msvcrt.get_osfhandle(fd))

    def lstat(self, parent: int, name: str) -> _Stat:
        h = _nt_open(parent, name, _FILE_READ_ATTRIBUTES, _FILE_OPEN, 0)
        try:
            _check_alias(h, name)
            return _stat_handle(h)
        finally:
            _close(h)

    def listdir(self, h: int) -> list[str]:
        buf = ctypes.create_string_buffer(64 * 1024)
        cls = _FileFullDirectoryRestartInfo
        names: list[str] = []
        while True:
            if not _GetFileInformationByHandleEx(h, cls, buf, len(buf)):
                err = ctypes.get_last_error()
                if err == _ERROR_NO_MORE_FILES:
                    return names
                raise ctypes.WinError(err)
            cls = _FileFullDirectoryInfo
            raw = buf.raw
            off = 0
            while True:
                # FILE_FULL_DIR_INFO: NextEntryOffset @0, FileNameLength @60, FileName @68.
                nxt = int.from_bytes(raw[off:off + 4], "little")
                nlen = int.from_bytes(raw[off + 60:off + 64], "little")
                name = raw[off + 68:off + 68 + nlen].decode("utf-16-le", "surrogatepass")
                if name not in (".", ".."):
                    names.append(name)
                if not nxt:
                    break
                off += nxt

    def open_read(self, parent: int, name: str) -> int:
        h = _nt_open(parent, name, _FILE_READ_DATA | _FILE_READ_ATTRIBUTES, _FILE_OPEN, _FILE_NON_DIRECTORY_FILE)
        try:
            _check_alias(h, name)
            return msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            _close(h)
            raise

    def create_temp(self, parent: int, name: str, mode: int) -> int:
        h = _nt_open(parent, name, _GENERIC_WRITE | _FILE_READ_ATTRIBUTES | _DELETE,
                     _FILE_CREATE, _FILE_NON_DIRECTORY_FILE)
        try:
            return msvcrt.open_osfhandle(h, os.O_WRONLY | os.O_BINARY)
        except BaseException:
            _close(h)
            raise

    def replace(self, parent: int, src: str, dst: str) -> None:
        h = _nt_open(parent, src, _DELETE | _FILE_READ_ATTRIBUTES, _FILE_OPEN, _FILE_NON_DIRECTORY_FILE)
        try:
            encoded = dst.encode("utf-16-le", "surrogatepass")
            offset = _FILE_RENAME_INFORMATION.FileName.offset
            size = max(ctypes.sizeof(_FILE_RENAME_INFORMATION), offset + len(encoded))
            buf = ctypes.create_string_buffer(size + ctypes.sizeof(wintypes.WCHAR))
            info = _FILE_RENAME_INFORMATION.from_buffer(buf)
            info.ReplaceIfExists = True
            info.RootDirectory = parent
            info.FileNameLength = len(encoded)
            ctypes.memmove(ctypes.addressof(buf) + offset, encoded, len(encoded))
            iosb = _IO_STATUS_BLOCK()
            status = _NtSetInformationFile(h, ctypes.byref(iosb), buf, size, _FileRenameInformation)
            if status < 0:
                raise _nt_error(status, dst)
        finally:
            _close(h)

    def mkdir(self, parent: int, name: str) -> None:
        _close(_nt_open(parent, name, _FILE_READ_DATA | _FILE_READ_ATTRIBUTES, _FILE_CREATE, _FILE_DIRECTORY_FILE))

    def unlink(self, parent: int, name: str) -> None:
        h = _nt_open(parent, name, _DELETE | _FILE_READ_ATTRIBUTES, _FILE_OPEN, 0)
        try:
            _check_alias(h, name)
            if _stat_handle(h).kind == "dir":
                raise IsADirectoryError(errno.EISDIR, "path is a folder", name)
            _set_delete(h, name)
        finally:
            _close(h)

    def rmdir(self, parent: int, name: str) -> None:
        h = _nt_open(parent, name, _DELETE | _FILE_READ_ATTRIBUTES, _FILE_OPEN, _FILE_DIRECTORY_FILE)
        try:
            _check_alias(h, name)
            _set_delete(h, name)
        finally:
            _close(h)
