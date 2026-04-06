import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
import pandas as pd 

#Pista suzuka
def generate_track():
    t = np.linspace(0, 2 *np.pi, 500)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    return x*10, y*5

def main():
    model = joblib.load(r'C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\src\models\models\model.pkl')
    X_track, y_track = generate_track()
    fig,ax = plt.subplots()
    ax.plot(X_track, y_track, 'gray')
    car, = ax.plot([],[],'ro')
    tyre_life = 1
    compound = 1
    lap_times = []
    position = 0
    
    def update(frame):
        nonlocal tyre_life, compound, lap_times, position
        
        #Features
        lap_time_mean = np.mean(lap_times[-3:]) if len(lap_times) >= 3 else 80
        lap_time_delta = lap_times[-1] - lap_times[-2] if len(lap_times) >= 2 else 0
        
        features = pd.DataFrame([[
            tyre_life, 
            compound, 
            lap_time_mean,
            lap_time_delta,
            tyre_life/50,
        ]], columns = ['TyreLife', 'Compound', 'lap_time_mean_3', 'lap_time_delta', 'tyre_ratio'])
        lap_time = model.predict(features)[0]
        lap_time.append(lap_time)
        
        #Velocidade proporcional
        speed = max(1, int(200 / lap_time))
        position = (position + speed) % len(X_track)
        car.set_data(X_track[position], y_track[position])
        tyre_life += 1
        return car,
    
    ani = FuncAnimation(fig, update, frames=500, blit=True, interval=50)
    
    plt.axis('equal')
    plt.show()
if __name__ == "__main__":
    main()        