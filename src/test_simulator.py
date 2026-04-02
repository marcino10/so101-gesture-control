from robot_controller import RobotController
import time

def test():
    # Will fail to connect to real port, will fall back to simulation
    with RobotController(port="/dev/ttyBOGUS") as r:
        print("Initialized controller.")
        if hasattr(r, 'visualizer') and r.visualizer:
            print("Visualizer successfully initialized.")
            
            for i in range(10):
                r.set_shoulder_pan(i * 10)
                time.sleep(0.1)
                
            print("Successfully updated visualization.")
        else:
            print("Warning: Visualizer did NOT initialize.")

if __name__ == "__main__":
    test()
