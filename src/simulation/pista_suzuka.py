import fastf1
def get_track():
    fastf1.Cache.enable_cache(r'C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\src\data\cache')
    session = fastf1.get_session(2023, 'Japan', 'Q')
    session.load()
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    x = pos['X'].values
    y = pos['Y'].values
    return x, y