import streamlit as st
import plotly.graph_objects as go
import numpy as np

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
    
    def display_bias_meter(self, bias_results):
        st.subheader("Bias Analysis")
        
        # Extract the overall bias score from results
        bias_score = bias_results.get("bias_score", 0)
        
        # Create three columns for layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display the gauge chart
            fig = self._create_gauge(bias_score)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display bias category and confidence
        bias_category = self._get_bias_category(bias_score)
        confidence = bias_results.get("confidence", 0.0)
        
        st.markdown(f"""
        **Bias Category:** {bias_category.replace('_', ' ').title()}  
        **Confidence:** {confidence:.1%}
        """)
        
        # Display key bias indicators
        if "bias_indicators" in bias_results:
            st.subheader("Key Bias Indicators")
            for indicator in bias_results["bias_indicators"]:
                st.markdown(f"- {indicator}")
