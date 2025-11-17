import os
import logging
import asyncio

from dotenv import load_dotenv
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ConversationHandler
)
# ----------
# Remove-Item -Recurse -Force faiss_index
# ----------
load_dotenv()  # загружаем .env из корня проекта

# Состояния
MAIN_MENU = 0
ASK_DB = 1
ASK_SUMMARY = 2

import indexer
from handlers import start_handler, menu_handler, ask_db_handler, ask_summary_handler, cancel_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set in environment variables.")
        return

    db_path = os.getenv("DB_PATH", "db.txt")
    collection = "default"
    col_dir = indexer._collection_dir(collection)
    if not os.path.isdir(col_dir):
        try:
            indexer.index_file(db_path, collection)
            logger.info(f"Database indexed from {db_path} into collection '{collection}'.")
        except Exception as e:
            logger.error(f"Failed to index database from {db_path}: {e}")


    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler('start', start_handler)],
        states={
            MAIN_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, menu_handler)],
            ASK_DB: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_db_handler)],
            ASK_SUMMARY:[
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_summary_handler),
                MessageHandler(filters.Document.ALL, ask_summary_handler)
            ],
        },
        fallbacks=[CommandHandler('start', start_handler), CommandHandler('cancel', cancel_handler)],
        allow_reentry=True,
    )
    app.add_handler(conv)
    logger.info("Bot is starting...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()