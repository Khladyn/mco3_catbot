import argparse
from cat_env import make_env
from training import train_bot
from utility import play_q_table

def play_env_with_cat(cat, render):
    # Train the agent ONCE. Use tuple unpacking (q_table, _)
    # since train_bot now returns the steps_data list as well.
    print(f"\nTraining agent against {cat} cat...")
    q_table, _ = train_bot(  # 🛑 Only train once, capture Q-table and discard steps_data
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
    print("\nDocumentation Pause: Window is active. Take a screenshot now.")
    input("Press Enter to close the window and exit the script...")
    # 🛑 END ALTERNATIVE CODE 🛑

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

    args = parser.parse_args()

    if args.cat != 'all-default':
        play_env_with_cat(args.cat, args.render)
    else:
        default_cats = ['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi']
        for cat in default_cats:
            play_env_with_cat(cat, args.render)

if __name__ == "__main__":
    main()