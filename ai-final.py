from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cv2
import os
from ultralytics import YOLO
import numpy as np

# Custom sentiment lexicon
custom_lexicon = {
    'Bohot badiya laga! Aise hi aur videos banao!': 2.0,
    'Dil khush kar diya, yaar!': 2.0,
    ' 👏': 2.0,
    'Mazaa aa gaya, ekdum zabardast!': 2.0,
    'Wah! Kya creativity hai, kamaal hai!': 2.0,
    'Aapki energy toh next level hai!': 2.0,
    'Kya mast content hai, bhai!': 2.0,
    ' ❤': 2.0,
    'Hamesha ki tarah, aapne dil jeet liya!': 2.0,
    'Itna amazing kaam kiya hai, proud of you!': 2.0,
    'Kya baat hai, ekdum hit content!': 1.5,
    'Full enjoy kar liya, has has ke pet dard ho gaya!': 1.5,
    '😂': 2.0,
    'Bilkul bekaar content hai, time waste.': -2.0,
    'Aapka content improve karne ki zaroorat hai.': -1.0,
    'Samajh nahi aaya, kya bakwaas tha yeh!': -1.5,
    'Kuch khaas nahi laga, better ho sakta tha.': -1.0,
    'Thoda boring tha, kuch naya lao.': -0.5,
    'Ye toh bilkul faltu tha!': -1.5,
    'Aapne kuch naya try kiya, lekin pasand nahi aaya.': -1.0,
    'Thoda aur mehnat karo, acha hoga.': -0.5,
    'Yeh kya bakwaas banaya hai, samajh nahi aaya.': -1.5,
    'Kaafi disappointing content tha.': -1.0,
    'This guy is in love again for sure': 1.5,
    'One of the best podcast, raw real full of life He\'s my dad\'s favourite actor': 2.0,
    '♥': 2.0,
    '🙌 ': 2.0,
    'I always like Ranveer\'s masti wala side': 2.0,
    '#beerbiceps ka every post bahut accha lagte hai': 2.0,
    'Absolutely delicious': 2.0,
    '😍': 2.0,
    'Iske shaadi mei barish nahi, puri bar ayegi.': 1.5,
    'Bhai, caption mein ingredients likh deta.': 1.0,
    'Pani me namak daldo, hogya boil chicken.': 1.0,
    'Saravana store annachi lady version maari irukinga.': 1.0,
    'Enadhu idhu ': 1.5,
    'haahaan': 1.5,
    '😂': 2.0,
    'Iyyoda': 2.0,
    'alagu thaaaa': 2.0,
    'Cute': 2.0,
    'Angel': 1.5,
    'Sky flying ': 2.0,
    'Azhge': 2.0,
    '🤍🤍 ': 2.0,
    'Kolukatta sapuduradhuku yedhuku da grwm uh': 0.0,
    'Sogam': 0.0,
    'hayyooo': 1.0,
    'Sapdurathuku makeup ah hayyooo': 0.0,
    '😢': -1.0,
    'ulti broo': 1.5,
    'Atta poochi': 2.0,
    'Aaai mathiri iruku': -1.5,
    'Dhosa dhosa than pizza pizza than': 1.0,
    'poi sollatha': 1.0,
    'Saala': 0.0,
    'ennaya Idhu': -1.0,
    'Pavangal': 1.0,
    ' fun❤': 2.0,
    'Purattasi': 1.0,
    'Karuthu': 1.0,
    'Bro unga vd tha super😍😍 nenga super chef.. unga voice la oru innocent irukku❤❤': 2.0,
    'Enna kandravii ithu': -1.0,
    'Chai': -2.0,
    'Kevalama iruki': -1.0,
    'un monjii Mari yee': -1.0,
    'Aathi': 0.0,
    'Paithiyam': -1.0,
    'Paithiyam ah? illa piravi paithiyam ah ?': -1.0,
    'Losu': -1.0
}

def initialize_sentiment_analyzer():
    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(custom_lexicon)
    return sid

def analyze_sentiment(comments, sid):
    positive_count = 0
    for comment in comments:
        score = sid.polarity_scores(comment)['compound']
        if score > 0:
            positive_count += 1
    return positive_count > len(comments) / 2  # Majority positive comments

def process_influencer_videos(root_dir, model_path, object_of_interest, frame_threshold=25, video_threshold=5, max_frames=1483):
    model = YOLO(model_path)
    sid = initialize_sentiment_analyzer()
    
    influencer_folders = [
        "Influencer_1_thatpotatoface",
        "Influencer_2_thrishakrishnaa",
        "Influencer_3_cookd",
        "Influencer_4_beersofbreakie",
        "Influencer_5_beerbiceps",
        "Influencer_6_rjmercyjohn"
    ]

    suitable_influencers = []
    
    for folder_name in influencer_folders:
        folder_path = os.path.join(root_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        print(f"\nProcessing folder: {folder_name}")
        videos_with_detections = 0

        for video_name in os.listdir(folder_path):
            if not video_name.endswith('.mp4'):
                continue

            video_path = os.path.join(folder_path, video_name)
            comment_file_path = video_path.replace('.mp4', '_comments.txt')

            if detect_in_video(video_path, model, object_of_interest, max_frames, frame_threshold):
                if os.path.isfile(comment_file_path):
                    with open(comment_file_path, 'r', encoding='utf-8') as f:
                        comments = f.readlines()
                    
                    if analyze_sentiment(comments, sid):
                        videos_with_detections += 1

            if videos_with_detections > video_threshold:
                suitable_influencers.append(folder_name)
                print(f"Folder '{folder_name}' meets the criteria for recommendation.")
                break
    
    if suitable_influencers:
        print("\nSuitable Influencers:")
        for influencer in suitable_influencers:
            print("\nFinal Recommendation:")
            print(f"Recommended Influencer: Folder '{influencer}' ")
    else:
        print("No suitable influencers found based on the criteria.")

def detect_in_video(video_path, model, object_of_interest, max_frames=1483, frame_threshold=25):
    vid_cap = cv2.VideoCapture(video_path)

    if not vid_cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
        return False

    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    detection_count = 0

    for count in frame_indices:
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, image = vid_cap.read()
        if not success:
            continue

        results = model(image)
        for obj in results[0].boxes:
            class_id = int(obj.cls)
            class_name = model.names[class_id]
            if class_name == object_of_interest:
                detection_count += 1

        if detection_count > frame_threshold:
            vid_cap.release()
            return True

    vid_cap.release()
    return False

if __name__ == "__main__":
    root_dir = r"E:\semesterfive\ai"
    model_path = r"E:\semesterfive\ai\insta_piyuu\runs\detect\train9\weights\best.pt"
    object_of_interest = input("Enter the object to detect (e.g., 'eyeliner', 'mic', 'pan'): ").strip()
    process_influencer_videos(root_dir, model_path, object_of_interest)
