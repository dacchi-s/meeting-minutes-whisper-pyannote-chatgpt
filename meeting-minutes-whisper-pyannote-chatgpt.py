import os
import subprocess
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import torch
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import tiktoken

def get_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value

load_dotenv()

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
HF_TOKEN = get_env_variable("HF_TOKEN")

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_to_wav(input_file):
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext in [".mp3", ".mp4", ".wma"]:
        print(f"Converting {input_file} to WAV")
        wav_file_path = input_file.replace(file_ext, ".wav")
        
        if file_ext == ".wma":
            try:
                subprocess.run(['ffmpeg', '-i', input_file, wav_file_path], check=True)
                print(f"Converted file saved as {wav_file_path}")
            except subprocess.CalledProcessError:
                raise ValueError(f"Error converting WMA file: {input_file}")
        else:
            audio = AudioSegment.from_file(input_file, format=file_ext[1:])
            audio.export(wav_file_path, format="wav")
            print(f"Converted file saved as {wav_file_path}")
        
        return wav_file_path
    elif file_ext == ".wav":
        return input_file
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def extract_audio_from_video(video_file):
    audio_file = video_file.replace(".mp4", ".wav")
    print(f"Extracting audio from {video_file}")
    try:
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(audio_file)
        video.close()
        audio.close()
        print(f"Audio extracted and saved as {audio_file}")
    except Exception as e:
        print(f"An error occurred while extracting audio: {str(e)}")
        raise
    return audio_file

def transcribe_audio(file_path, model_name, max_chunk_size=600):
    print(f"Transcribing audio from {file_path} using model {model_name}")
    device = get_device()
    model = whisper.load_model(model_name, device=device)
    audio = AudioSegment.from_wav(file_path)
    duration = len(audio) / 1000  # Duration in seconds
    chunk_size = min(max_chunk_size, duration) 

    transcriptions = []
    segments = []

    for i in range(0, int(duration), int(chunk_size)):
        chunk = audio[i*1000:(i+int(chunk_size))*1000]
        chunk_path = f"temp_chunk_{i}.wav"
        try:
            chunk.export(chunk_path, format="wav")
            result = model.transcribe(chunk_path)
            transcriptions.append(result["text"])
            for segment in result["segments"]:
                segment["start"] += i
                segment["end"] += i
                segments.append(segment)
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    transcription = " ".join(transcriptions)
    print(f"Transcription completed")
    return transcription, segments

def diarize_audio(file_path, num_speakers=None, min_speakers=None, max_speakers=None):
    print(f"Performing speaker diarization on {file_path}")
    device = get_device()
    print(f"Using device: {device}")
    
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        pipeline.to(device)
    except Exception as e:
        print(f"Failed to load the pipeline: {e}")
        return None
    
    if num_speakers is not None:
        diarization = pipeline(file_path, num_speakers=num_speakers)
    elif min_speakers is not None and max_speakers is not None:
        diarization = pipeline(file_path, min_speakers=min_speakers, max_speakers=max_speakers)
    else:
        diarization = pipeline(file_path)

    print(f"Speaker diarization completed: {diarization}")
    return diarization

def assign_speakers_to_segments(segments, diarization):
    if diarization is None:
        print("Diarization is None. Skipping speaker assignment.")
        return []

    print("Assigning speakers to segments")
    speaker_segments = []
    current_speaker = None
    current_segment = None
    unmatched_segments = []

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        matched = False

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap = min(end_time, turn.end) - max(start_time, turn.start)
            if overlap > 0:
                if current_speaker == speaker:
                    # Append to the current segment
                    current_segment['end'] = end_time
                    current_segment['text'] += " " + text
                else:
                    # Save the current segment and start a new one
                    if current_segment:
                        speaker_segments.append(current_segment)
                    current_speaker = speaker
                    current_segment = {
                        "start": start_time,
                        "end": end_time,
                        "speaker": speaker,
                        "text": text
                    }
                matched = True
                break
        
        if not matched:
            print(f"No matching speaker found for segment: {text} [{start_time} - {end_time}]")
            unmatched_segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": "UNKNOWN",
                "text": text
            })

    if current_segment:
        speaker_segments.append(current_segment)

    # Merge unmatched segments into speaker_segments
    all_segments = speaker_segments + unmatched_segments
    all_segments.sort(key=lambda x: x['start'])  # Sort by start time

    print("Speakers assigned to segments")
    return all_segments

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    return len(encoding.encode(text))

def split_transcript(transcript, max_tokens_per_chunk):
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for word in words:
        word_tokens = count_tokens(word)
        if current_chunk_tokens + word_tokens > max_tokens_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def merge_minutes(minutes_list):
    """複数の議事録をマージして一つの整形された議事録にする"""
    final_minutes = {
        "概要": [],
        "主要な議題と決定事項": [],
        "アクションアイテム": [],
        "参加者の主な発言": [],
        "次回に向けての課題": []
    }
    
    for minutes in minutes_list:
        sections = minutes.split("■")
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if "概要" in section:
                content = section.split("概要", 1)[1].strip()
                if content and content not in final_minutes["概要"]:
                    final_minutes["概要"].append(content)
            
            elif "主要な議題と決定事項" in section:
                content = section.split("主要な議題と決定事項", 1)[1].strip()
                if content:
                    items = [item.strip() for item in content.split("\n") if item.strip()]
                    for item in items:
                        if item not in final_minutes["主要な議題と決定事項"]:
                            final_minutes["主要な議題と決定事項"].append(item)
            
            elif "アクションアイテム" in section:
                content = section.split("アクションアイテム", 1)[1].strip()
                if content:
                    items = [item.strip() for item in content.split("\n") if item.strip() and item.startswith("-")]
                    for item in items:
                        if item not in final_minutes["アクションアイテム"]:
                            final_minutes["アクションアイテム"].append(item)
            
            elif "参加者の主な発言" in section:
                content = section.split("参加者の主な発言", 1)[1].strip()
                if content:
                    items = [item.strip() for item in content.split("\n") if item.strip()]
                    for item in items:
                        if item not in final_minutes["参加者の主な発言"]:
                            final_minutes["参加者の主な発言"].append(item)
            
            elif "次回に向けての課題" in section:
                content = section.split("次回に向けての課題", 1)[1].strip()
                if content:
                    items = [item.strip() for item in content.split("\n") if item.strip() and item.startswith("-")]
                    for item in items:
                        if item not in final_minutes["次回に向けての課題"]:
                            final_minutes["次回に向けての課題"].append(item)
    
    # 最終的な議事録の形式に整形
    formatted_minutes = []
    formatted_minutes.append("■ 概要")
    formatted_minutes.append("\n".join(final_minutes["概要"]))
    
    formatted_minutes.append("\n\n■ 主要な議題と決定事項")
    formatted_minutes.append("\n".join(final_minutes["主要な議題と決定事項"]))
    
    formatted_minutes.append("\n\n■ アクションアイテム")
    formatted_minutes.append("\n".join(final_minutes["アクションアイテム"]))
    
    formatted_minutes.append("\n\n■ 参加者の主な発言")
    formatted_minutes.append("\n".join(final_minutes["参加者の主な発言"]))
    
    formatted_minutes.append("\n\n■ 次回に向けての課題")
    formatted_minutes.append("\n".join(final_minutes["次回に向けての課題"]))
    
    return "\n".join(formatted_minutes)

def generate_minutes(speaker_segments, model="gpt-4o-mini"):
    """会話の書き起こしから議事録を生成する"""
    print(f"Generating meeting minutes using OpenAI API with model: {model}")
    client = OpenAI(api_key=OPENAI_API_KEY)

    transcript = "\n".join([f"Speaker {seg['speaker']} [{seg['start']} - {seg['end']}]: {seg['text']}" 
                           for seg in speaker_segments])
    max_tokens_per_chunk = 4096
    chunks = split_transcript(transcript, max_tokens_per_chunk)
    all_minutes = []

    system_prompt = """以下の会話の議事録を作成してください。

要件：
1. 重要な議題、決定事項、アクションアイテムを明確に抽出してください
2. 各話者の主要な発言や貢献を簡潔にまとめてください
3. 時系列順に整理し、議論の流れが分かるようにしてください
4. 技術的な用語や専門用語は保持してください
5. 冗長な表現や重複した内容は省略してください

出力フォーマット：
■ 概要
[会議の全体的な要約]

■ 主要な議題と決定事項
1. [議題1]
   - 決定事項：
   - 議論のポイント：

2. [議題2]
   ...

■ アクションアイテム
- [担当者/チーム]: [タスク内容] (期限がある場合は記載)

■ 参加者の主な発言
- Speaker XXXX:
  - [重要な発言内容]
  
■ 次回に向けての課題
- [課題1]
- [課題2]
"""

    for i, chunk in enumerate(chunks):
        context = ""
        if i > 0:
            context += "（前のセッションの続き）\n\n"
        if i < len(chunks) - 1:
            context += "\n\n（次のセッションに続く）"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context + chunk}
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=4096,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )

                if response.choices and len(response.choices) > 0:
                    minutes = response.choices[0].message.content.strip()
                    if not any(section in minutes for section in ["概要", "主要な議題", "アクションアイテム"]):
                        raise ValueError("Invalid minutes format received")
                    all_minutes.append(minutes)
                    break
                else:
                    print("No choices in the response")
                    if attempt == max_retries - 1:
                        return None
            except Exception as e:
                print(f"Error generating minutes for chunk {i} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                continue

    # すべてのチャンクの議事録をマージ
    final_minutes = merge_minutes(all_minutes)
    return final_minutes

def parse_arguments():
    parser = argparse.ArgumentParser(description="Speech to Text with Speaker Diarization")
    parser.add_argument("audio_file", help="Path to the audio or video file")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        default="base", help="Whisper model to use")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use for generating minutes (default: gpt-4o-mini)")
    parser.add_argument("--num_speakers", type=int, help="Number of speakers (if known)")
    parser.add_argument("--min_speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max_speakers", type=int, help="Maximum number of speakers")
    return parser.parse_args()

def main(args):
    print(f"Processing file: {args.audio_file}")
    file_ext = os.path.splitext(args.audio_file)[1].lower()
    if file_ext == ".mp4":
        audio_file = extract_audio_from_video(args.audio_file)
    else:
        audio_file = args.audio_file
    
    if file_ext in [".mp3", ".mp4", ".wav"]:
        audio_file = convert_to_wav(audio_file)
    
    transcription, segments = transcribe_audio(audio_file, args.model)
    print(f"Transcription: {transcription}")
    diarization = diarize_audio(audio_file, num_speakers=args.num_speakers, 
                                min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    speaker_segments = assign_speakers_to_segments(segments, diarization)

    output_file = os.path.splitext(audio_file)[0] + "_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Speakers assigned to segments\n")
        for segment in speaker_segments:
            segment_text = f"Speaker {segment['speaker']} [{segment['start']} - {segment['end']}]: {segment['text']}"
            print(segment_text)
            f.write(segment_text + "\n")
    
    print(f"Results written to {output_file}")

    minutes = generate_minutes(speaker_segments, model=args.openai_model)
    if minutes is not None:
        minutes_file = os.path.splitext(audio_file)[0] + "_minutes.txt"
        with open(minutes_file, 'w', encoding='utf-8') as f:
            f.write(minutes)
        print(f"Meeting minutes written to {minutes_file}")
    else:
        print("Failed to generate meeting minutes.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)