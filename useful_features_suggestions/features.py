# 1 Разделяй видео и аудио:
# ffmpeg -i input.mp4 -vn output_audio.mp3
# ffmpeg -i input.mp4 -an output_video.mp4

# 2 Добавляй собственную аудиодорожку
# ffmpeg -i input_video.mp4 -i custom_audio.mp3 -c:v copy -c:a aac output.mp4

# Модуль 4: Текстовый поиск по видео
# Позволяет пользователю искать слова в транскрибации.
# Подсвечивает найденные слова и указывает временные метки.
# Фронтенд:

# Используй текстовый редактор с подсветкой (например, CodeMirror).
# @app.get("/search/")
# async def search_word(word: str, transcript: str):
#     positions = [i for i, w in enumerate(transcript.split()) if w == word]
#     return {"positions": positions}


# Модуль 3: Создание клипов
# Функция API вызывает твой метод для генерации клипов на основе LLM:
# Передача транскрибации → Возвращает временные метки для клипов.
# Нарезка видео:
# Используй moviepy или ffmpeg для вырезки фрагментов.
# from moviepy.video.io.VideoFileClip import VideoFileClip

# def cut_clips(video_path, timestamps):
#     clips = []
#     video = VideoFileClip(video_path)
#     for start, end in timestamps:
#         clip = video.subclip(start, end)
#         clips.append(clip)
#     final_clip = concatenate_videoclips(clips)
#     final_clip.write_videofile("output/clips.mp4")


# Основные модули сервиса
# Модуль 1: Загрузка и выгрузка видео
# Загрузка:
# Принимает видеофайл через API (используй FastAPI с UploadFile).
# Сохраняет видео на сервере/облаке.
# Вывод итогового видео:
# Применение ffmpeg для обработки и экспорта видео без потерь качества.
# Реализация на FastAPI:

# from fastapi import FastAPI, UploadFile
# import shutil

# app = FastAPI()

# @app.post("/upload/")
# async def upload_video(file: UploadFile):
#     with open(f"videos/{file.filename}", "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return {"filename": file.filename}

# @app.get("/download/{filename}")
# async def download_video(filename: str):
#     return FileResponse(f"videos/{filename}")
