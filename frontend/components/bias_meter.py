import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict

class BiasAnalyzer:
    def __init__(self):
        self.bias_categories = {
            "far_left": (-1.0, -0.6),
            "left": (-0.6, -0.2),
            "center": (-0.2, 0.2),
            "right": (0.2, 0.6),
            "far_right": (0.6, 1.0)
        }
        
        self.category_colors = {
            "far_left": "#FF0000",
            "left": "#FF6B6B",
            "center": "#4CAF50",
            "right": "#6B8EFF",
            "far_right": "#0000FF"
        }
    
    def _create_gauge(self, bias_score):
        # Normalize bias score to be between -1 and 1
        bias_score = max(min(bias_score, 1.0), -1.0)
        
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=(bias_score + 1) * 50,  # Convert from [-1,1] to [0,100]
            gauge={
                'axis': {'range': [0, 100],
                        'ticktext': ['Far Left', 'Left', 'Center', 'Right', 'Far Right'],
                        'tickvals': [10, 30, 50, 70, 90]},
                'bar': {'color': "darkgray"},
                'steps': [
                    {'range': [0, 20], 'color': self.category_colors["far_left"]},
                    {'range': [20, 40], 'color': self.category_colors["left"]},
                    {'range': [40, 60], 'color': self.category_colors["center"]},
                    {'range': [60, 80], 'color': self.category_colors["right"]},
                    {'range': [80, 100], 'color': self.category_colors["far_right"]}
                ],
            },
            title={'text': "Political Bias Meter"}
        ))
        
        # Update layout
        fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=30, b=30),
            font={'size': 14}
        )
        
        return fig
    
    def _get_bias_category(self, bias_score):
        for category, (low, high) in self.bias_categories.items():
            if low <= bias_score < high:
                return category
        return "far_right" if bias_score >= 1.0 else "far_left"
    
    def _get_bias_color(self, bias_score: float) -> str:
        """Get color for bias score visualization"""
        if bias_score <= -0.6:
            return 'rgb(65, 105, 225)'  # Strong Left (Blue)
        elif bias_score <= -0.2:
            return 'rgb(100, 149, 237)'  # Moderate Left (Light Blue)
        elif bias_score <= 0.2:
            return 'rgb(169, 169, 169)'  # Center (Gray)
        elif bias_score <= 0.6:
            return 'rgb(205, 92, 92)'  # Moderate Right (Light Red)
        else:
            return 'rgb(178, 34, 34)'  # Strong Right (Red)
            
    def display_bias_meter(self, bias_score: float):
        """Display the bias meter visualization"""
        st.subheader("⚖️ Bias Analysis")
        
        # Ensure bias_score is a float and within bounds
        try:
            bias_score = float(bias_score)
            bias_score = max(-1.0, min(1.0, bias_score))
        except (TypeError, ValueError):
            bias_score = 0.0
            
        # Create the meter visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bias_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {
                    'range': [-1, 1],
                    'tickwidth': 1,
                    'ticktext': ['Far Left', 'Left', 'Center', 'Right', 'Far Right'],
                    'tickvals': [-0.8, -0.4, 0, 0.4, 0.8]
                },
                'bar': {'color': self._get_bias_color(bias_score)},
                'steps': [
                    {'range': [-1, -0.6], 'color': 'rgba(65, 105, 225, 0.3)'},  # Strong Left
                    {'range': [-0.6, -0.2], 'color': 'rgba(100, 149, 237, 0.3)'},  # Moderate Left
                    {'range': [-0.2, 0.2], 'color': 'rgba(169, 169, 169, 0.3)'},  # Center
                    {'range': [0.2, 0.6], 'color': 'rgba(205, 92, 92, 0.3)'},  # Moderate Right
                    {'range': [0.6, 1], 'color': 'rgba(178, 34, 34, 0.3)'}  # Strong Right
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': bias_score
                }
            },
            title = {'text': "Political Bias Meter", 'font': {'size': 24, 'color': 'white'}}
        ))
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=70, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 16}
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Display bias category with emoji
        bias_category = self._get_bias_category(bias_score)
        category_emojis = {
            "far_left": "⬅️",
            "left": "↖️",
            "center": "⬆️",
            "right": "↗️",
            "far_right": "➡️"
        }
        
        st.markdown(
            f"### {category_emojis.get(bias_category, '🎯')} Bias Category: "
            f"_{bias_category.replace('_', ' ').title()}_"
        )
        
        # Display key bias indicators
        if "bias_indicators" in bias_results:
            st.subheader("Key Bias Indicators")
            for indicator in bias_results["bias_indicators"]:
                st.markdown(f"- {indicator}")
