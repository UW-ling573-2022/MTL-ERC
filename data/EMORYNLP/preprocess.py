import json
import pandas as pd


def main(input_file, output_file):
    f = open(input_file)

    friends_data = dict(
                        Utterance=[],
                        Speaker=[],
                        Emotion=[],
                        Dialogue_ID=[],
                        Utterance_ID=[],
                        )

    # read season from json file
    season = json.load(f)

    i = 1
    # read each episode
    for episode in season['episodes']:
        episode_id = episode['episode_id']

        # read each scene
        for scene in episode['scenes']:
            scene_id = scene['scene_id']

            # read each utterance
            for utterance in scene['utterances']:
                utterance_id = utterance['utterance_id']
                speaker = utterance['speakers'][0] if utterance['speakers'] else 'unknown'
                friends_data['Dialogue_ID'].append(int(scene_id.split('_')[-1][1:]))
                friends_data['Utterance_ID'].append(int(utterance_id.split('_')[-1][1:]) - 1)
                friends_data['Speaker'].append(speaker.split()[0])
                friends_data['Emotion'].append(utterance['emotion'])
                friends_data['Utterance'].append(utterance['transcript'])

    friends_df = pd.DataFrame(friends_data)

    # save data frame to .tsv
    friends_df.to_csv(output_file, sep=',', index=False)

    print('File saved!')
    # show sample
    friends_df.head()


if __name__ == "__main__":
    main("raw/emotion-detection-dev.json", "dev.csv")
    main("raw/emotion-detection-trn.json", "train.csv")
    main("raw/emotion-detection-tst.json", "test.csv")
