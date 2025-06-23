from app.game.env import LShapedGridWorldEnv
from app.game.gui import LShapedGridWorldGUI

if __name__ == "__main__":
    env = LShapedGridWorldEnv()
    gui = LShapedGridWorldGUI(env)
    gui.run()
