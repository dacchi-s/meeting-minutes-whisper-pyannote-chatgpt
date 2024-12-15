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

def translate_text(text, target_language, model="gpt-4o-mini"):
    """Translate text using the OpenAI API"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = f"Translate the following text to {target_language}. Maintain the original meaning and nuance."
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model=model,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def process_transcription(transcription, segments, output_language, detected_language=None):
    """Process transcription and translate if necessary"""
    if not detected_language:
        # More accurate language detection
        japanese_chars = len([c for c in transcription if ord(c) > 0x3040 and ord(c) < 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF])
        total_chars = len(transcription)
        detected_language = "japanese" if japanese_chars / total_chars > 0.1 else "english"
    
    need_translation = detected_language != output_language
    
    if need_translation:
        print(f"Translating from {detected_language} to {output_language}")
        
        # Translation considering chunk size constraints
        max_chunk_size = 4000  # Consider OpenAI API limits
        translated_segments = []
        current_chunk = []
        current_length = 0
        
        for i, segment in enumerate(segments):
            current_chunk.append((i, segment['text']))
            current_length += len(segment['text'])
            
            if current_length >= max_chunk_size or i == len(segments) - 1:
                # Translate current chunk
                chunk_text = "\n\n".join([f"[{idx}] {text}" for idx, text in current_chunk])
                translated_chunk = translate_text(chunk_text, output_language)
                
                if translated_chunk:
                    # Distribute translated text to corresponding segments
                    chunk_parts = translated_chunk.split("\n\n")
                    for j, (idx, _) in enumerate(current_chunk):
                        if j < len(chunk_parts):
                            translated_part = chunk_parts[j].split("] ", 1)[1] if "] " in chunk_parts[j] else chunk_parts[j]
                            segments[idx]['original_text'] = segments[idx]['text']
                            segments[idx]['text'] = translated_part
                
                # Reset chunk
                current_chunk = []
                current_length = 0
    
    return segments

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

def merge_minutes(minutes_list, output_language="japanese"):
    """Merge multiple meeting minutes into a single formatted document"""
    # Set section names based on language
    if output_language == "japanese":
        sections = {
            "概要": [],
            "主要な議題と決定事項": [],
            "アクションアイテム": [],
            "参加者の主な発言": [],
            "次回に向けての課題": []
        }
    else:
        sections = {
            "Overview": [],
            "Key Topics and Decisions": [],
            "Action Items": [],
            "Key Statements from Participants": [],
            "Issues for Next Meeting": []
        }
    
    for minutes in minutes_list:
        sections_split = minutes.split("■")
        for section in sections_split:
            section = section.strip()
            if not section:
                continue
            
            for key in sections.keys():
                if key in section:
                    content = section.split(key, 1)[1].strip()
                    if content:
                        items = [item.strip() for item in content.split("\n") if item.strip()]
                        for item in items:
                            # For action items and issues, only add items that start with "-"
                            if (key in ["アクションアイテム", "次回に向けての課題", "Action Items", "Issues for Next Meeting"]):
                                if item.startswith("-") and item not in sections[key]:
                                    sections[key].append(item)
                            # For other sections, add all non-empty items
                            elif item not in sections[key]:
                                sections[key].append(item)
    
    # Format the final minutes
    formatted_minutes = []
    
    # Add each section with its content
    for key, items in sections.items():
        formatted_minutes.append(f"■ {key}")
        if items:
            formatted_minutes.append("\n".join(items))
        else:
            # Add placeholder text for empty sections
            placeholder = "該当なし" if output_language == "japanese" else "None"
            formatted_minutes.append(placeholder)
        formatted_minutes.append("")  # Add empty line between sections
    
    return "\n".join(formatted_minutes)

def get_minutes_prompt(language):
    """Return the minutes generation prompt based on language"""
    if language == "japanese":
        return """以下の会話の議事録を作成してください。

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
    else:
        return """Create meeting minutes for the following conversation.

Requirements:
1. Clearly extract important topics, decisions, and action items
2. Summarize key statements and contributions from each speaker concisely
3. Organize chronologically to show the flow of discussion
4. Maintain technical terms and specialized vocabulary
5. Omit redundant expressions and duplicate content

Output Format:
■ Overview
[Overall meeting summary]

■ Key Topics and Decisions
1. [Topic 1]
   - Decision:
   - Discussion Points:

2. [Topic 2]
   ...

■ Action Items
- [Person/Team]: [Task] (Include deadline if specified)

■ Key Statements from Participants
- Speaker XXXX:
  - [Important statements]

■ Issues for Next Meeting
- [Issue 1]
- [Issue 2]
"""

def generate_minutes(speaker_segments, output_language="japanese", model="gpt-4o-mini"):
    """Generate meeting minutes from speaker segments"""
    print(f"Generating meeting minutes in {output_language} using OpenAI API with model: {model}")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare transcript with speaker information
    transcript = "\n".join([f"Speaker {seg['speaker']} [{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}" 
                           for seg in speaker_segments])
    max_tokens_per_chunk = 4096  # Adjust based on model limits
    chunks = split_transcript(transcript, max_tokens_per_chunk)
    all_minutes = []

    # Get language-specific prompt
    system_prompt = get_minutes_prompt(output_language)

    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Add context information for better continuity
        context = ""
        if i > 0:
            # Include last part of previous chunk
            prev_context = chunks[i-1].split("\n")[-3:]  # Last 3 lines
            context += "Previous context:\n" + "\n".join(prev_context) + "\n\n"
            
        if i < len(chunks) - 1:
            # Include beginning of next chunk
            next_context = chunks[i+1].split("\n")[:3]  # First 3 lines
            context += "\n\nNext context:\n" + "\n".join(next_context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nCurrent segment:\n{chunk}"}
        ]

        # Retry logic for API calls
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
                    # Verify required sections based on language
                    expected_sections = ["概要", "主要な議題", "アクションアイテム"] if output_language == "japanese" \
                        else ["Overview", "Key Topics", "Action Items"]
                    
                    if not any(section in minutes for section in expected_sections):
                        raise ValueError("Invalid minutes format received")
                    
                    all_minutes.append(minutes)
                    break
                else:
                    print("No response choices received from API")
                    if attempt == max_retries - 1:
                        return None
            except Exception as e:
                print(f"Error generating minutes for chunk {i} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                continue

    # Merge all chunks into final minutes
    if all_minutes:
        final_minutes = merge_minutes(all_minutes, output_language)
        return final_minutes
    
    return None

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
    parser.add_argument("--output_language", choices=["japanese", "english"], default="japanese",
                        help="Language for the output (transcription and minutes)")
    parser.add_argument("--detect_language", action="store_true",
                        help="Automatically detect input language using Whisper")
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
    processed_segments = process_transcription(transcription, segments, args.output_language)
    
    diarization = diarize_audio(audio_file, num_speakers=args.num_speakers, 
                               min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    speaker_segments = assign_speakers_to_segments(processed_segments, diarization)

    # Output transcription (includes original text if translation exists)
    output_file = os.path.splitext(audio_file)[0] + f"_output_{args.output_language}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription in {args.output_language}\n")
        for segment in speaker_segments:
            segment_text = f"Speaker {segment['speaker']} [{segment['start']} - {segment['end']}]: {segment['text']}"
            if 'original_text' in segment:
                segment_text += f"\nOriginal: {segment['original_text']}"
            print(segment_text)
            f.write(segment_text + "\n")
    
    print(f"Results written to {output_file}")

    # Generate meeting minutes
    minutes = generate_minutes(speaker_segments, output_language=args.output_language, model=args.openai_model)
    if minutes is not None:
        minutes_file = os.path.splitext(audio_file)[0] + f"_minutes_{args.output_language}.txt"
        with open(minutes_file, 'w', encoding='utf-8') as f:
            f.write(minutes)
        print(f"Meeting minutes written to {minutes_file}")
    else:
        print("Failed to generate meeting minutes.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
