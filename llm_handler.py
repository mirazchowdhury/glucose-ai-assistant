from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",  # or "llama3-8b-8192" for faster responses
            temperature=0.3
        )
        self.setup_prompts()
    
    def setup_prompts(self):
        """Define specialized prompts for different agents"""
        
        # Health Advisor prompts
        self.health_advisor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable health advisor AI specializing in diabetes and glucose management. 
            Your role is to analyze glucose data and provide helpful, evidence-based recommendations.
            
            IMPORTANT DISCLAIMERS:
            - Always remind users to consult healthcare professionals
            - Never provide definitive medical diagnoses
            - Focus on lifestyle and monitoring suggestions
            - Be supportive and encouraging
            
            Analyze the provided glucose data and give personalized recommendations."""),
            ("human", "Glucose data: {glucose_data}\nPatient context: {patient_context}\nProvide analysis and recommendations.")
        ])
        
        # Pattern Analyzer prompts
        self.pattern_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data pattern recognition expert specializing in glucose trends.
            Analyze glucose readings to identify:
            - Trends (increasing, decreasing, stable)
            - Patterns (dawn phenomenon, post-meal spikes)
            - Anomalies (unusual readings)
            - Risk indicators
            
            Provide clear, structured analysis."""),
            ("human", "Analyze these glucose readings: {readings}\nTime period: {time_period}")
        ])
        
        # Alert System prompts
        self.alert_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an alert system for glucose monitoring.
            Evaluate glucose readings for:
            - Hypoglycemia (< 70 mg/dL)
            - Hyperglycemia (> 200 mg/dL)
            - Rapid changes
            - Concerning patterns
            
            Provide immediate actionable alerts when necessary."""),
            ("human", "Current glucose: {current_glucose}\nRecent readings: {recent_readings}\nPatient profile: {patient_profile}")
        ])
    
    def generate_health_advice(self, glucose_data, patient_context=""):
        """Generate personalized health recommendations"""
        chain = LLMChain(llm=self.llm, prompt=self.health_advisor_prompt)
        
        response = chain.run(
            glucose_data=glucose_data,
            patient_context=patient_context
        )
        return response
    
    def analyze_patterns(self, readings, time_period=""):
        """Analyze glucose patterns"""
        chain = LLMChain(llm=self.llm, prompt=self.pattern_prompt)
        
        response = chain.run(
            readings=readings,
            time_period=time_period
        )
        return response
    
    def check_alerts(self, current_glucose, recent_readings, patient_profile=""):
        """Check for critical alerts"""
        chain = LLMChain(llm=self.llm, prompt=self.alert_prompt)
        
        response = chain.run(
            current_glucose=current_glucose,
            recent_readings=recent_readings,
            patient_profile=patient_profile
        )
        return response