# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/chess_server.py

Chess MCP server — deterministic referee for LLM-vs-LLM chess matches.
Runs as a stdio subprocess of Claude Code (or whichever CLI the gamemaster
agent uses). Owns per-game state JSON files under CLAWMEETS_CHESS_STATE_DIR.

The gamemaster agent calls these tools to validate moves and publishes the
returned `state_json_content` into the chatroom files via `update_file`, so
every participant — and any local browser viewer pointed at `board.html` —
sees the same canonical board.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ENV_VAR = "CLAWMEETS_CHESS_STATE_DIR"


def _state_dir() -> Path:
    p = os.environ.get(ENV_VAR)
    if not p:
        raise RuntimeError(
            f"{ENV_VAR} is not set. The chess MCP server is expected to be "
            f"launched by the clawmeets runner, which sets this via the "
            f"mcps/chess/mcp.json launch spec."
        )
    d = Path(p).resolve()
    (d / "games").mkdir(parents=True, exist_ok=True)
    return d


def _game_dir(game_id: str) -> Path:
    # Validate the id so a malicious caller can't escape the games/ root.
    if not game_id or "/" in game_id or ".." in game_id or "\\" in game_id:
        raise ValueError(f"Invalid game_id: {game_id!r}")
    d = _state_dir() / "games" / game_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path(game_id: str) -> Path:
    return _game_dir(game_id) / "board_state.json"


def _view_html_source() -> str:
    return (Path(__file__).parent / "chess_view.html").read_text(encoding="utf-8")


def _state_js_wrapper(state: dict) -> str:
    # The board viewer loads state via <script src="./board_state.js"> to
    # sidestep Chrome's file:// CORS rule on fetch(). This wrapper assigns
    # the state dict onto window.__CHESS_STATE so the viewer can read it
    # after the script loads.
    return "window.__CHESS_STATE = " + json.dumps(state, indent=2) + ";\n"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(game_id: str) -> dict:
    p = _state_path(game_id)
    if not p.exists():
        raise FileNotFoundError(f"No such game: {game_id!r}. Use start_game first.")
    return json.loads(p.read_text())


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def _save_state(state: dict) -> None:
    """Write the canonical state JSON + the JS wrapper that the viewer loads.

    The MCP owns the viewer files (board.html + board_state.js) so the
    gamemaster never has to remember to publish them — whatever moves happen,
    the browser sees them on the next poll tick. The canonical board_state.json
    is also written here; the gamemaster still publishes a copy of that JSON
    into the chatroom so white/black/narrator can read the position.
    """
    game_id = state["game_id"]
    gd = _game_dir(game_id)
    _atomic_write(gd / "board_state.json", json.dumps(state, indent=2))
    _atomic_write(gd / "board_state.js", _state_js_wrapper(state))
    # Write board.html only if it's missing — it's a static viewer page per
    # game. Avoid rewriting on every move to keep file mtimes sane for any
    # external watchers.
    html_path = gd / "board.html"
    if not html_path.exists():
        _atomic_write(html_path, _view_html_source())


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc
    try:
        import chess
        import chess.pgn
    except ImportError as exc:
        raise RuntimeError(
            "python-chess is required. Try: pip install --upgrade clawmeets "
            "(>=1.1.8 ships python-chess as a baseline dependency)."
        ) from exc

    mcp = FastMCP("clawmeets-chess")

    def _replay(move_history: list[dict]) -> "chess.Board":
        board = chess.Board()
        for mv in move_history:
            board.push(chess.Move.from_uci(mv["uci"]))
        return board

    def _compute_result(board: "chess.Board") -> tuple[Optional[str], Optional[str]]:
        if board.is_checkmate():
            return ("0-1" if board.turn == chess.WHITE else "1-0"), "checkmate"
        if board.is_stalemate():
            return "1/2-1/2", "stalemate"
        if board.is_insufficient_material():
            return "1/2-1/2", "insufficient_material"
        if board.is_seventyfive_moves():
            return "1/2-1/2", "seventyfive_moves"
        if board.is_fivefold_repetition():
            return "1/2-1/2", "fivefold_repetition"
        return None, None

    def _build_pgn(
        game_id: str,
        white_name: str,
        black_name: str,
        started_at: str,
        move_history: list[dict],
        result: Optional[str],
    ) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = f"ClawMeets — {game_id}"
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Date"] = started_at[:10].replace("-", ".")
        game.headers["Result"] = result or "*"
        node = game
        for mv in move_history:
            node = node.add_variation(chess.Move.from_uci(mv["uci"]))
        return str(game)

    def _state_from(
        board: "chess.Board",
        game_id: str,
        white_name: str,
        black_name: str,
        started_at: str,
        move_history: list[dict],
        narration: Optional[list[dict]] = None,
    ) -> dict:
        result, result_reason = _compute_result(board)
        is_over = result is not None
        return {
            "game_id": game_id,
            "white_name": white_name,
            "black_name": black_name,
            "fen": board.fen(),
            "pgn": _build_pgn(game_id, white_name, black_name, started_at, move_history, result),
            "active_color": None if is_over else ("white" if board.turn == chess.WHITE else "black"),
            "in_check": board.is_check() and not is_over,
            "move_number": board.fullmove_number,
            "move_history": move_history,
            "narration": list(narration) if narration else [],
            "is_game_over": is_over,
            "result": result,
            "result_reason": result_reason,
            "started_at": started_at,
            "updated_at": _now_iso(),
        }

    @mcp.tool()
    def start_game(game_id: str, white_name: str = "white", black_name: str = "black") -> dict:
        """Create a new chess game at the starting position.

        Overwrites any existing state files for this game_id. The MCP writes
        `board.html`, `board_state.json`, and `board_state.js` directly to its
        state directory — the gamemaster does NOT need to publish those for
        the viewer to work. `view_path` in the return is the absolute path the
        user opens in a browser. The gamemaster SHOULD still publish
        `state_json_content` as `board_state.json` into the chatroom so the
        player agents (white, black, narrator) can read the position.
        """
        board = chess.Board()
        started_at = _now_iso()
        state = _state_from(board, game_id, white_name, black_name, started_at, [])
        _save_state(state)
        return {
            "ok": True,
            "game_id": game_id,
            "fen": state["fen"],
            "active_color": state["active_color"],
            "view_path": str(_game_dir(game_id) / "board.html"),
            "state_json_content": json.dumps(state, indent=2),
        }

    @mcp.tool()
    def make_move(game_id: str, color: str, move: str) -> dict:
        """Apply a move to the game.

        `color` must be "white" or "black" and must match whose turn it is —
        the MCP rejects off-turn moves without mutating state.
        `move` accepts SAN ("e4", "Nf3", "O-O", "exd5", "e8=Q") or UCI
        ("e2e4", "g1f3", "e7e8q").

        On success returns {ok: true, san, uci, fen, move_number, active_color,
        is_game_over, result, result_reason, state_json_content}.
        On rejection returns {ok: false, error}.
        """
        color_norm = color.lower().strip()
        if color_norm not in ("white", "black"):
            return {"ok": False, "error": f"color must be 'white' or 'black', got {color!r}"}
        try:
            state = _load_state(game_id)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        if state["is_game_over"]:
            return {"ok": False, "error": f"game is already over ({state['result']} — {state['result_reason']})"}

        board = _replay(state["move_history"])
        expected = "white" if board.turn == chess.WHITE else "black"
        if color_norm != expected:
            return {"ok": False, "error": f"not {color_norm}'s turn — it's {expected}'s move"}

        move_str = move.strip()
        parsed = None
        san_err = None
        try:
            parsed = board.parse_san(move_str)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError) as e:
            san_err = str(e) or type(e).__name__
        if parsed is None:
            try:
                parsed = board.parse_uci(move_str)
            except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError) as e:
                uci_err = str(e) or type(e).__name__
                return {
                    "ok": False,
                    "error": (
                        f"could not parse move {move_str!r} as SAN ({san_err}) or UCI ({uci_err}). "
                        "Use SAN like 'Nf3' or UCI like 'g1f3'."
                    ),
                }

        ply = len(state["move_history"]) + 1
        san = board.san(parsed)
        uci = parsed.uci()
        board.push(parsed)

        new_history = list(state["move_history"]) + [{
            "san": san, "uci": uci, "color": color_norm, "ply": ply,
        }]
        new_state = _state_from(
            board, game_id, state["white_name"], state["black_name"],
            state["started_at"], new_history,
            state.get("narration", []),
        )
        _save_state(new_state)
        return {
            "ok": True,
            "san": san,
            "uci": uci,
            "fen": new_state["fen"],
            "move_number": new_state["move_number"],
            "active_color": new_state["active_color"],
            "in_check": new_state["in_check"],
            "is_game_over": new_state["is_game_over"],
            "result": new_state["result"],
            "result_reason": new_state["result_reason"],
            "state_json_content": json.dumps(new_state, indent=2),
        }

    @mcp.tool()
    def get_state(game_id: str) -> dict:
        """Return the full canonical state for a game."""
        return _load_state(game_id)

    @mcp.tool()
    def get_legal_moves(game_id: str, from_square: Optional[str] = None) -> list[str]:
        """Legal moves (SAN) in the current position.

        Optionally filter by source square (e.g. 'e2', 'g1'). Returns an empty
        list if the game is over or the square is invalid.
        """
        state = _load_state(game_id)
        if state["is_game_over"]:
            return []
        board = _replay(state["move_history"])
        moves = list(board.legal_moves)
        if from_square:
            try:
                sq = chess.parse_square(from_square.lower().strip())
            except ValueError:
                return []
            moves = [m for m in moves if m.from_square == sq]
        return [board.san(m) for m in moves]

    @mcp.tool()
    def resign(game_id: str, color: str) -> dict:
        """End the game by resignation. `color` is the side that resigns."""
        color_norm = color.lower().strip()
        if color_norm not in ("white", "black"):
            return {"ok": False, "error": f"color must be 'white' or 'black', got {color!r}"}
        try:
            state = _load_state(game_id)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        if state["is_game_over"]:
            return {"ok": False, "error": f"game is already over ({state['result']})"}
        state["is_game_over"] = True
        state["result"] = "0-1" if color_norm == "white" else "1-0"
        state["result_reason"] = "resignation"
        state["active_color"] = None
        state["updated_at"] = _now_iso()
        # Refresh PGN with the new result header.
        state["pgn"] = _build_pgn(
            game_id, state["white_name"], state["black_name"],
            state["started_at"], state["move_history"], state["result"],
        )
        _save_state(state)
        return {
            "ok": True,
            "result": state["result"],
            "result_reason": state["result_reason"],
            "state_json_content": json.dumps(state, indent=2),
        }

    @mcp.tool()
    def post_narration(game_id: str, text: str) -> dict:
        """Append a narrator comment to the game.

        The comment attaches to the most recently played ply (or ply 0 if no
        moves have been made yet). Rewrites board_state.json and board_state.js
        so the local viewer picks it up on its next poll tick.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "text is required"}
        if len(text) > 1000:
            return {"ok": False, "error": "text too long (max 1000 chars)"}
        try:
            state = _load_state(game_id)
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        history = state.get("move_history", [])
        last = history[-1] if history else None
        entry = {
            "ply": last["ply"] if last else 0,
            "move_number": ((last["ply"] - 1) // 2 + 1) if last else 0,
            "color": last["color"] if last else None,
            "text": text,
            "at": _now_iso(),
        }
        state.setdefault("narration", []).append(entry)
        state["updated_at"] = _now_iso()
        _save_state(state)
        return {"ok": True, "entry": entry}

    @mcp.tool()
    def list_games() -> list[str]:
        """Enumerate all game ids with state files in this MCP's state directory."""
        games_dir = _state_dir() / "games"
        if not games_dir.exists():
            return []
        return sorted(
            p.name for p in games_dir.iterdir()
            if p.is_dir() and (p / "board_state.json").exists()
        )

    mcp.run()


if __name__ == "__main__":
    main()
