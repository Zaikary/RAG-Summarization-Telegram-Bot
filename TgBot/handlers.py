from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes
import logging
import re
import os
import tempfile

import indexer
import llm_handler

logger = logging.getLogger(__name__)

MAIN_MENU_TEXT = "Выберите действие:"
BTN_DB = "📚 Работа с БД"
BTN_SUMMARY = "📝 Суммаризация текста"
MAIN_MENU_KB = ReplyKeyboardMarkup(
    [[BTN_DB, BTN_SUMMARY]], resize_keyboard=True, one_time_keyboard=True
    )

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello, " + MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
    logger.info(f"......started the bot........")
    return 0

async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    logger.info(f"User selected menu option: {text}")
    if text == BTN_DB:
        await update.message.reply_text("RAG, работа с базой ", reply_markup=ReplyKeyboardRemove())
        return 1
    if text == BTN_SUMMARY:
        await update.message.reply_text("Суммаризация текста", reply_markup=ReplyKeyboardRemove())
        return 2
    await update.message.reply_text("Нужно выбрать" + MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
    return 0

async def extract_text_from_file(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf8") as f:
            return f.read()
    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join([page.page_content for page in pages])
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

async def ask_db_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    logger.info(f"User asked a DB question: {question}")
    results = indexer.search_hybrid(question, collection="default", k=5)
    if not results:
        answer = "Ничего не найдено по вашему запросу."
    else:
        best = results[0]
        # answer = f"Ближайший фрагмент:\n\n{best['content']}"
        context = best['content']

        try:
            answer = llm_handler.generate_answer_from_context(question, context)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback: просто показываем найденный фрагмент
            answer = f"📋 Найдено в базе:\n\n{context}"

    await update.message.reply_text(answer)
    await update.message.reply_text(MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
    return 0

async def ask_summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        document = update.message.document
        file_name = document.file_name
        logger.info(f"User uploaded file: {file_name}")

        _, ext = os.path.splitext(file_name)
        if ext.lower() not in ['.txt', '.pdf']:
            await update.message.reply_text(
                "⚠️ Поддерживаются только: .txt, .pdf"
            )
            await update.message.reply_text(MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
            return 0

        await update.message.reply_text("⏳ Загружаю файл...")

        try:
            file = await document.get_file()

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_path = tmp_file.name

            await file.download_to_drive(tmp_path)
            logger.info(f"File downloaded to: {tmp_path}")

            await update.message.chat.send_action(action="typing")
            text = await extract_text_from_file(tmp_path)

            os.unlink(tmp_path)

            if len(text.strip()) < 50:
                await update.message.reply_text("⚠️ Файл слишком короткий.")
                await update.message.reply_text(MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
                return 0

            logger.info(f"Extracted {len(text)} chars from file")

        except Exception as e:
            logger.error(f"File processing error: {e}")
            await update.message.reply_text(f"❌ Ошибка обработки файла: {e}")
            await update.message.reply_text(MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
            return 0

    else:
        text = (update.message.text or "").strip()
        logger.info(f"User provided text: {len(text)} chars")

        if not text:
            await update.message.reply_text("Отправьте текст или файл.")
            return 2

        if len(text) < 100:
            await update.message.reply_text(
                "⚠️ Текст слишком короткий (минимум 100 символов)."
            )
            await update.message.reply_text(MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
            return 0

    await update.message.chat.send_action(action="typing")

    try:
        summary = llm_handler.summarize_text(text)
        await update.message.reply_text(summary, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        fallback = text[:500] + "..." if len(text) > 500 else text
        await update.message.reply_text(
            f"⚠️ Не удалось создать краткое содержание.\n\n"
            f"Первые 500 символов:\n{fallback}"
        )

    await update.message.reply_text(MAIN_MENU_TEXT, reply_markup=MAIN_MENU_KB)
    return 0

async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Действие отменено.", reply_markup=MAIN_MENU_KB)
    logger.info("User cancelled the action.")
    return 0
