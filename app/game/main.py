from app.game.env import FrozenLLakeEnv
from app.game.gui import FrozenLLakeGUI

if __name__ == "__main__":
    env = FrozenLLakeEnv()
    gui = FrozenLLakeGUI(env)
    gui.run()
