import asyncio
import threading
import time
import numpy as np
import pyaudio
import wave
import os
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from dotenv import load_dotenv
from pydub import AudioSegment

# Load token from .env
load_dotenv()
token = os.getenv("BOT_TOKEN")
bot = Bot(token=token)
dp = Dispatcher()

# Audio config
THRESHOLD = 20
RECORD_SECONDS = 10

# Store threads and stop events per chat
_listening_threads: dict[int, threading.Thread] = {}
_stop_events: dict[int, threading.Event] = {}
_selected_devices: dict[int, int] = {}

# Get available audio input devices
def get_audio_devices():
    print("🎤 Available Audio Input Devices:")
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append((i, info["name"]))
            print(f"[{i}] {info['name']}")
    p.terminate()
    return devices

# Send inline buttons to select mic
@dp.message(Command("start_listening"))
async def start_listening_handler(message: Message):
    chat_id = message.chat.id
    if chat_id in _listening_threads:
        await message.answer("⚠️ Already listening.")
        return

    devices = get_audio_devices()
    if not devices:
        await message.answer("❌ No input devices found.")
        return

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"{idx}: {name}", callback_data=f"mic_{idx}")]
        for idx, name in devices
    ])
    await message.answer("🎙️ Choose your input device:", reply_markup=keyboard)

@dp.callback_query(lambda c: c.data.startswith("mic_"))
async def select_microphone_handler(callback_query: CallbackQuery):
    chat_id = callback_query.message.chat.id
    device_index = int(callback_query.data.split("_")[1])

    _selected_devices[chat_id] = device_index
    await bot.answer_callback_query(callback_query.id, f"🎤 Selected device {device_index}")

    stop_event = threading.Event()
    _stop_events[chat_id] = stop_event
    loop = asyncio.get_running_loop()
    thread = threading.Thread(
        target=listen_and_record,
        args=(chat_id, stop_event, loop),
        daemon=True
    )
    _listening_threads[chat_id] = thread
    thread.start()
    await bot.send_message(chat_id, f"✅ Started listening using device index {device_index}. Send /stop_listening to stop.")

def listen_and_record(chat_id: int, stop_event: threading.Event, loop: asyncio.AbstractEventLoop):
    device_index = _selected_devices.get(chat_id)
    if device_index is None:
        print(f"[ERROR] No device selected for chat_id: {chat_id}")
        return

    print(f"[DEBUG] Listener thread started for chat_id: {chat_id} using device {device_index}")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=1024
    )

    try:
        while not stop_event.is_set():
            data = stream.read(1024, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data ** 2))

            if rms > THRESHOLD:
                print("[DEBUG] Trigger detected!")
                frames = [data]
                for _ in range(int(44100 / 1024 * RECORD_SECONDS)):
                    if stop_event.is_set():
                        break
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(data)

                filename = f"recording_{chat_id}_{int(time.time())}.wav"
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(44100)
                    wf.writeframes(b''.join(frames))

                print(f"[DEBUG] Saved file: {filename}")
                asyncio.run_coroutine_threadsafe(send_and_delete(chat_id, filename), loop)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f"[DEBUG] Listener thread ended for chat_id: {chat_id}")

async def send_and_delete(chat_id: int, filename: str):
    try:
        mp3_filename = filename.replace(".wav", ".mp3")
        sound = AudioSegment.from_wav(filename)
        sound.export(mp3_filename, format="mp3")
        await bot.send_message(chat_id, "🎙️ Detected sound! Here's your recording:")
        file = FSInputFile(mp3_filename)
        await bot.send_document(chat_id, file)
    finally:
        for f in [filename, mp3_filename]:
            if os.path.exists(f):
                os.remove(f)

@dp.message(Command("stop_listening"))
async def stop_listening_handler(message: Message):
    chat_id = message.chat.id
    stop_event = _stop_events.get(chat_id)
    if stop_event:
        stop_event.set()
        thread = _listening_threads.pop(chat_id, None)
        if thread:
            print(f"[DEBUG] Waiting for thread to end for chat_id: {chat_id}")
            thread.join()
        _stop_events.pop(chat_id, None)
        await message.answer("🛑 Stopped listening.")
    else:
        await message.answer("⚠️ I'm not listening right now.")

async def main():
    print("🚀 Bot is running...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
