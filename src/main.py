import asyncio
import logging
import sys

from app import Application
from settings import AppSettings, ValidationError


async def main() -> None:
    try:
        settings = AppSettings()
    except ValidationError as e:
        logging.error(e)
        sys.exit(1)

    try:
        app = Application(settings)
        await app.start()
    except asyncio.CancelledError:
        logging.warning("Cancelled")
    except Exception:
        logging.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
