import plotly.express as px
import plotly.graph_objects as go

def plot_privacy_tradeoff(df):
    """
    Plots the privacy-utility tradeoff (Epsilon vs Accuracy).
    """
    fig = px.line(df, x='Epsilon', y='Accuracy', markers=True, 
                  title='Privacy-Utility Tradeoff',
                  labels={'Epsilon': 'Privacy Budget (ε)', 'Accuracy': 'Model Accuracy'})
    fig.update_traces(line_color='#FF4B4B', marker=dict(size=8))
    fig.update_layout(template='plotly_dark')
    return fig

def plot_training_curves(df):
    """
    Plots training loss and accuracy over epochs.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Loss'], mode='lines+markers', name='Loss', line=dict(color='#0068C9')))
    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Accuracy'], mode='lines+markers', name='Accuracy', line=dict(color='#83C9FF')))
    fig.update_layout(title='Training Curves', xaxis_title='Epoch', yaxis_title='Value', template='plotly_dark')
    return fig
