"""Process entrypoint. Render sets PORT; local runs default to 8000."""

import os

import uvicorn

from gateway.app import create_app
from gateway.config import get_settings
from gateway.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
app = create_app(settings)


def main() -> None:
    port = int(os.environ.get("PORT", str(settings.port)))
    host = os.environ.get("HOST", settings.host)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
