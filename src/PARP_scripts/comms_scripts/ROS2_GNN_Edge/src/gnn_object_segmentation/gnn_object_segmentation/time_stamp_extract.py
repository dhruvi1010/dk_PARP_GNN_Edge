import sqlite3
import json

def extract_timestamps_from_db3(db3_file, topic_name, output_file):
    """
    Extract timestamps from SQLite .db3 file and save as JSON.
    
    Args:
        db3_file (str): Path to the .db3 file.
        topic_name (str): Name of the topic to extract timestamps for.
        output_file (str): JSON file to save the timestamps.
    """
    timestamps = {}

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db3_file)
        cursor = conn.cursor()

        # Get the topic ID for the radar topic
        cursor.execute("SELECT id FROM topics WHERE name = ?;", (topic_name,))
        topic_result = cursor.fetchone()

        if not topic_result:
            print(f"Topic {topic_name} not found in {db3_file}")
            conn.close()
            return

        topic_id = topic_result[0]

        # Get all message timestamps for this topic
        try:
            cursor.execute("SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp;", (topic_id,))
            message_results = cursor.fetchall()
        except Exception:
            print("Error querying messages table")
            conn.close()
            return

        # Process results
        for i, (timestamp,) in enumerate(message_results):
            timestamps[i] = timestamp / 1e9  # Convert from nanoseconds to seconds

        conn.close()

        # Save timestamps to a JSON file
        with open(output_file, 'w') as f:
            json.dump(timestamps, f, indent=4)
        print(f"Extracted {len(timestamps)} timestamps from {db3_file} and saved to {output_file}")

    except Exception as e:
        print(f"Error extracting timestamps from {db3_file}: {e}")

# Example usage
extract_timestamps_from_db3(
    db3_file='/home/flw-6gem-dev/dev/RoboFUSE_Dataset/CPPS_Static_Scenario/CPPS_Vertical/Robot_1/rosbag/20250219_171846/20250219_171846_0.db3',
    topic_name='/ep03/vicon/pose',
    output_file='timestamps_ep03_vicon.json'
)

extract_timestamps_from_db3(
    db3_file='/home/flw-6gem-dev/dev/RoboFUSE_Dataset/CPPS_Static_Scenario/CPPS_Vertical/Robot_2/rosbag/20250219_171851/20250219_171851_0.db3',
    topic_name='/ep05/vicon/pose',
    output_file='timestamps_ep05_vicon.json'
)
