from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm


def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, required=True, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--ex-path', type=str, required=True, help='location to save the exported data')

    args = parser.parse_args()
    event_data = event_accumulator.EventAccumulator(args.in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        # print(key)
        if key != 'train_reward_one2five_time':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
            print(df)
    print(df)
    df.to_csv(args.ex_path)

    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    main()


