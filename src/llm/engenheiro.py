import requests

def explain_decision(data):
    prompt = f"""
    Voce é um engenheiro de corrida de F1 e precisa explicar a decisão de pit stop para a equipe.
    Dados atuais:
    - Volta : {data['lap']}
    - Vida do pneu: {data['tyre_life']}
    - Tempo médio das ultimas 3 voltas: {data['lap_time_mean_3']}
    - Delta do tempo da ultima volta: {data['lap_time_delta']}
    - Pit stop necessário: {data['pit_stop_needed']}
    - Tempo previsto para o pit stop: {data['predicted_pit_time']}
    - Tempo previsto para a volta: {data['predicted_lap_time']}
    Explique de forma profissional e clara para a equipe. Focando que nosso piloto ganhe a corrida.print
    """
    
    response = requests.post(
        'http://localhost:11434/api/generate',
        json = {
            'model' : 'qwen3:8b',
            'prompt' : prompt,
            'stream' : False,
        }
    )
    return response.json().get('response', 'Sem resposta')