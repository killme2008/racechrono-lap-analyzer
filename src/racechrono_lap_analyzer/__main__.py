"""Allow running the app with: python -m racechrono_lap_analyzer"""

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main():
    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--server.headless=true"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
