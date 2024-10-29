import cv2
import os

def extract_frames(video_path, output_folder, num_frames=200):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the video
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    interval = int(total_frames / num_frames)  # Interval to capture frames
    
    # Extract frames at each interval
    frame_count = 0
    saved_frames = 0
    
    while saved_frames < num_frames and frame_count < total_frames:
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # Read the frame
        success, frame = video.read()
        if not success:
            break
        
        # Save the frame as a high-quality PNG
        filename = os.path.join(output_folder, f"frame_{saved_frames + 1}.png")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 compression for best quality
        
        saved_frames += 1
        frame_count += interval
    
    video.release()
    print(f"Saved {saved_frames} frames in {output_folder}")

# Example usage
extract_frames("path/to/video.mp4", "output_frames_folder")


