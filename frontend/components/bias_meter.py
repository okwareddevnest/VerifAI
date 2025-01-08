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
        st.subheader("⚖️ Political Bias Analysis")
        
        # Ensure bias_score is a float and within bounds
        try:
            bias_score = float(bias_score)
            bias_score = max(-1.0, min(1.0, bias_score))
        except (TypeError, ValueError):
            bias_score = 0.0
            
        # Define colors for gradient effect
        colors = {
            'far_left': '#1E3FCC',    # Deep Blue
            'left': '#4169E1',        # Royal Blue
            'center': '#808080',      # Gray
            'right': '#CD5C5C',       # Indian Red
            'far_right': '#8B0000'    # Dark Red
        }
        
        # Create dynamic color based on bias score
        def get_gradient_color(score):
            if score <= -0.6:
                return colors['far_left']
            elif score <= -0.2:
                return colors['left']
            elif score <= 0.2:
                return colors['center']
            elif score <= 0.6:
                return colors['right']
            else:
                return colors['far_right']
        
        # Create the meter visualization with animation
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = bias_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': 0, 'position': 'top'},
            gauge = {
                'axis': {
                    'range': [-1, 1],
                    'tickwidth': 3,
                    'tickcolor': "white",
                    'ticktext': ['Far Left', 'Left', 'Center', 'Right', 'Far Right'],
                    'tickvals': [-0.8, -0.4, 0, 0.4, 0.8],
                    'tickfont': {'color': 'white', 'size': 14}
                },
                'bar': {
                    'color': get_gradient_color(bias_score),
                    'thickness': 0.75
                },
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [-1, -0.6], 'color': f"{colors['far_left']}30"},
                    {'range': [-0.6, -0.2], 'color': f"{colors['left']}30"},
                    {'range': [-0.2, 0.2], 'color': f"{colors['center']}30"},
                    {'range': [0.2, 0.6], 'color': f"{colors['right']}30"},
                    {'range': [0.6, 1], 'color': f"{colors['far_right']}30"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 5},
                    'thickness': 0.8,
                    'value': bias_score
                }
            },
            title = {
                'text': "Political Bias Meter",
                'font': {'size': 24, 'color': 'white'},
                'align': 'center'
            },
            number = {
                'font': {'size': 50, 'color': get_gradient_color(bias_score)},
                'prefix': "",
                'suffix': ""
            }
        ))
        
        # Add dynamic animations and styling
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=70, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 16},
            annotations=[
                dict(
                    text=f"Bias Score: {bias_score:.2f}",
                    x=0.5,
                    y=0.1,
                    showarrow=False,
                    font={'size': 20, 'color': get_gradient_color(bias_score)}
                )
            ]
        )
        
        # Add animation frames
        fig.update_traces(
            selector=dict(type='indicator'),
            delta_increasing={'color': colors['far_right']},
            delta_decreasing={'color': colors['far_left']},
            animatescale=True
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Display bias category with emoji and dynamic styling
        bias_category = self._get_bias_category(bias_score)
        category_emojis = {
            "far_left": "⬅️",
            "left": "↖️",
            "center": "⬆️",
            "right": "↗️",
            "far_right": "➡️"
        }
        
        # Create dynamic category display
        st.markdown(
            f"""
            <div style='
                text-align: center;
                padding: 20px;
                border-radius: 10px;
                background-color: {get_gradient_color(bias_score)}20;
                border: 2px solid {get_gradient_color(bias_score)};
                margin: 10px 0;
            '>
                <h2 style='color: {get_gradient_color(bias_score)}; margin: 0;'>
                    {category_emojis.get(bias_category, '🎯')} 
                    {bias_category.replace('_', ' ').title()}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Add interpretation
        interpretations = {
            "far_left": "Strong liberal/progressive bias detected",
            "left": "Moderate liberal bias detected",
            "center": "Relatively balanced perspective",
            "right": "Moderate conservative bias detected",
            "far_right": "Strong conservative bias detected"
        }
        
        st.markdown(
            f"""
            <div style='
                text-align: center;
                font-style: italic;
                color: #B0B0B0;
                margin-top: 10px;
            '>
                {interpretations.get(bias_category, "Interpretation unavailable")}
            </div>
            """,
            unsafe_allow_html=True
        )

