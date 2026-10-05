import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config_from_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the general knowledge config JSON file.",
    )
    args, _ = parser.parse_known_args(argv)
    return args.config, load_config(args.config)
