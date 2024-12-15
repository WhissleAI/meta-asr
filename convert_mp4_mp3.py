from moviepy import VideoFileClip
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

def mp4_to_mp3_chunks(file_path, chunk_duration=5000):
    """
    Converts an MP4 video file to small chunks of MP3 audio format.

    Args:
        file_path (str): Path to the MP4 video file.
        chunk_duration (int): Duration of each chunk in milliseconds. Default is 5000ms (5 seconds).

    Returns:
        generator: Yields BytesIO objects of MP3 chunks.
    """
    # Load video and extract audio
    video = VideoFileClip(file_path)
    audio_path = "temp_audio.wav"

    # Write audio to a temporary WAV file
    audio = video.audio
    audio.write_audiofile(audio_path, codec="pcm_s16le")

    # Convert WAV audio to MP3 using pydub
    audio_segment = AudioSegment.from_file(audio_path, format="wav")

    # Split audio into chunks and yield as BytesIO
    for i in range(0, len(audio_segment), chunk_duration):
        chunk = audio_segment[i:i + chunk_duration]
        chunk_buffer = BytesIO()
        chunk.export(chunk_buffer, format="mp3")
        chunk_buffer.seek(0)
        yield chunk_buffer

# Example usage:
if __name__ == "__main__":
    video_path = "batch_processing/real_world/11_Steps_To_Impress_In_Any_Panel_Discussion_Media_Training.mp4"  # Replace with your video file path
    for idx, audio_chunk in enumerate(mp4_to_mp3_chunks(video_path)):
        print(f"Playing chunk {idx + 1}")
        chunk_audio = AudioSegment.from_file(audio_chunk, format="mp3")
        play(chunk_audio)
