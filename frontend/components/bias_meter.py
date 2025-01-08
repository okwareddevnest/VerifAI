import streamlit as st
import plotly.graph_objects as go

class BiasAnalyzer:
    def __init__(self):
        """Initialize the bias analyzer"""
        pass
        
    def _get_bias_color(self, bias_score: float) -> str:
        """Get color based on bias score"""
        if bias_score <= -0.6:
            return "rgba(30, 63, 204, 0.9)"  # Deep blue for far left
        elif bias_score <= -0.2:
            return "rgba(86, 119, 252, 0.9)"  # Light blue for left
        elif bias_score <= 0.2:
            return "rgba(128, 128, 128, 0.9)"  # Gray for center
        elif bias_score <= 0.6:
            return "rgba(252, 86, 86, 0.9)"  # Light red for right
        else:
            return "rgba(204, 30, 30, 0.9)"  # Deep red for far right
            
    def _get_bias_category(self, bias_score: float) -> str:
        """Get bias category based on score"""
        if bias_score <= -0.6:
            return "Far Left"
        elif bias_score <= -0.2:
            return "Left"
        elif bias_score <= 0.2:
            return "Center"
        elif bias_score <= 0.6:
            return "Right"
        else:
            return "Far Right"
            
    def display_bias_meter(self, bias_score: float):
        """Display bias meter visualization"""
        try:
            # Ensure bias_score is a float and within bounds
            bias_score = float(bias_score)
            bias_score = max(-1.0, min(1.0, bias_score))
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = bias_score * 100,  # Convert to percentage
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Political Bias", 'font': {'size': 24}},
                gauge = {
                    'axis': {
                        'range': [-100, 100],
                        'tickwidth': 1,
                        'tickcolor': "darkgray",
                        'ticktext': ["Far Left", "Left", "Center", "Right", "Far Right"],
                        'tickvals': [-80, -40, 0, 40, 80]
                    },
                    'bar': {'color': self._get_bias_color(bias_score)},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-100, -60], 'color': "rgba(30, 63, 204, 0.2)"},
                        {'range': [-60, -20], 'color': "rgba(86, 119, 252, 0.2)"},
                        {'range': [-20, 20], 'color': "rgba(128, 128, 128, 0.2)"},
                        {'range': [20, 60], 'color': "rgba(252, 86, 86, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(204, 30, 30, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': bias_score * 100
                    }
                }
            ))
            
            # Update layout
            fig.update_layout(
                paper_bgcolor = "white",
                font = {'color': "darkgray", 'family': "Arial"}
            )
            
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Display category
            category = self._get_bias_category(bias_score)
            st.markdown(f"### Political Leaning: {category}")
            
        except Exception as e:
            st.error(f"Error displaying bias meter: {str(e)}")
            st.markdown("### Political Leaning: Unable to determine")

