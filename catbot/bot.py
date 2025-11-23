import argparse
import csv
from cat_env import make_env
from training import train_bot
from utility import play_q_table
from datetime import datetime

FILE_PATH = 'C:\\Users\\David-Mini\\Desktop\\DLSU\\CSC613M\\MCO3\\mco3\\release'

def generate_csv(stats):
    # 🛑 CHANGE FILE PATH 🛑
    with open(f'{FILE_PATH}\\catbot-{datetime.now().strftime("%Y%m%d%H%M%S")}.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        stats_header = ['cat', 'episode', 'success count', 'success rate', 'average no. of steps', 'average exploration count', 'average exploration rate', 'average exploitation count', 'average exploitation rate', 'epsilon decay']
        writer.writerow(stats_header)

        for cat_keys in stats.keys():
            for eps_keys in stats[cat_keys].keys():
                cat_ep_data = [cat_keys, eps_keys]
                combined_list = cat_ep_data +stats[cat_keys][eps_keys]
                writer.writerow(combined_list)
                

def play_env_with_cat(cat, render, pause_allowed=True):
    # Train the agent ONCE. Use tuple unpacking (q_table, _)
    # since train_bot now returns the steps_data list as well.
    print(f"\nTraining agent against {cat} cat...")
    q_table, _, bot_data = train_bot(  # 🛑 Only train once, capture Q-table and discard steps_data
        cat_name=cat,
        render=render
    )

    # Play using the trained Q-table
    env = make_env(cat_type=cat)

    # 🛑 CAPTURE THE FINAL STEP COUNT 🛑
    final_steps_taken = play_q_table(env, q_table, max_steps=60, window_title=f'Cat Chase - Final Trained Bot - {cat}')

    # 🛑 UPDATE THE PRINT STATEMENT 🛑
    print("\nTraining complete! Starting game with trained bot...")

    # Informative message about the final run
    print(f"Final Game Result: Cat was caught in {final_steps_taken} steps (Max steps: 60).")

    # 🛑 ALTERNATIVE INTERACTIVE PAUSE 🛑
    if pause_allowed == True:
        print("\nDocumentation Pause: Window is active. Take a screenshot now.")
        input("Press Enter to close the window and exit the script...")
    # 🛑 END ALTERNATIVE CODE 🛑

    return bot_data

def main():
    parser = argparse.ArgumentParser(description='Train and play Cat Chase bot')
    parser.add_argument('--cat',
                        choices=['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi', 'trainer', 'spidercat', 'cheddar', 'pumpkinpie', 'milky', 'taro', 'all-default'],
                        default='batmeow',
                        help='Type of cat to train against (default: mittens)')
    parser.add_argument('--render',
                        type=int,
                        default=100,
                        help='Render the environment every n episodes (default: -1, no rendering)')
    parser.add_argument('--pause',
                        type=bool,
                        default=False,
                        help='Add pause before ending the training environment (default: True, pausing after training)')
    parser.add_argument('--csv',
                        type=bool,
                        default=False,
                        help='Get csv stats at the end of the run (default: False, no csv generation)')

    args = parser.parse_args()

    stats = {}

    if args.cat != 'all-default':
        stats[args.cat] = play_env_with_cat(args.cat, args.render, args.pause)
    else:
        default_cats = ['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi']
        for cat in default_cats:
            stats[cat] = play_env_with_cat(cat, args.render, args.pause)

    if args.csv == True:
        generate_csv(stats)

if __name__ == "__main__":
    main()