import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
from abc import ABC, abstractmethod

class DataProcessor:
    """Handles all data loading and preprocessing operations"""
    
    @staticmethod
    def load_csv(file_path):
        """Load CSV file into pandas DataFrame"""
        try:
            df = pd.read_csv(file_path)
            # Apply cleaning right after loading
            return DataProcessor.clean_data(df)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def clean_data(df):
        """Clean and prepare data"""
        
        # FIX 1: Strip whitespace from all column names
        df.columns = df.columns.str.strip()
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Fill missing values
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0)
        df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
        
        # Clean department names
        df['Department'] = df['Department'].fillna('Unknown').str.strip()
        
        # === FIX FOR MISSING PUBLICATION_TYPE ===
        # Impute publication_type based on journal_name if column doesn't exist
        if 'publication_type' not in df.columns:
            df['publication_type'] = df['journal_name'].apply(DataProcessor._derive_pub_type)
            
        return df

    @staticmethod
    def _derive_pub_type(journal_name):
        """Helper to guess publication type from journal name"""
        if pd.isna(journal_name):
            return "Journal Article" # Default
        
        txt = str(journal_name).lower()
        
        # Heuristic keywords for classification
        conference_keywords = ['conference', 'proceeding', 'symposium', 'workshop', 'congress', 'ieee', 'acm', 'seminar']
        book_keywords = ['handbook', 'book series', 'springer', 'chapter']
        
        if any(kw in txt for kw in conference_keywords):
            return "Conference Paper"
        elif any(kw in txt for kw in book_keywords):
            return "Book Chapter"
        else:
            return "Journal Article"

class ResearchDomainExtractor:
    """Extracts research domains from publication titles"""
    
    DOMAIN_KEYWORDS = {
        'Machine Learning': ['machine learning', 'deep learning', 'neural network', 'cnn', 'lstm', 'classification', 'prediction', 'ai', 'artificial intelligence'],
        'Data Mining': ['data mining', 'clustering', 'association', 'frequent pattern', 'apriori'],
        'IoT & Sensors': ['iot', 'sensor', 'wireless sensor', 'wsn', 'smart', 'rfid', 'zigbee'],
        'Security & Cryptography': ['security', 'cryptography', 'authentication', 'malware', 'encryption', 'blockchain', 'cyber', 'attack', 'privacy'],
        'Network & Communication': ['network', 'routing', 'protocol', 'communication', 'vanet', 'manet', 'mobile', 'wireless', '5g', '4g'],
        'Image Processing': ['image', 'steganography', 'watermarking', 'segmentation', 'vision', 'face'],
        'Cloud Computing': ['cloud', 'fog computing', 'edge computing', 'grid'],
        'Software Engineering': ['software', 'refactoring', 'code smell', 'fault prediction', 'testing'],
        'Big Data': ['big data', 'hadoop', 'mapreduce', 'analytics'],
        'Healthcare': ['health', 'medical', 'disease', 'patient', 'cancer', 'tumor', 'cardiac']
    }
    
    @staticmethod
    def extract_domain(title):
        """Extract research domain from publication title"""
        if pd.isna(title):
            return 'Other'
        
        title_lower = title.lower()
        matches = []
        
        for domain, keywords in ResearchDomainExtractor.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in title_lower:
                    matches.append(domain)
                    break
        
        return matches[0] if matches else 'Other'


class Visualizer(ABC):
    """Abstract base class for all visualizations"""
    
    @abstractmethod
    def create(self, data, **kwargs):
        """Create visualization"""
        pass

class BarChartVisualizer(Visualizer):
    """Creates bar charts"""
    
    def create(self, data, x, y, title, color=None, orientation='v'):
        fig = px.bar(
            data, 
            x=x, 
            y=y, 
            title=title,
            color=color,
            orientation=orientation,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            height=500,
            hovermode='x unified',
            font=dict(size=12)
        )
        return fig

class PieChartVisualizer(Visualizer):
    """Creates pie charts"""
    
    def create(self, data, values, names, title):
        fig = px.pie(
            data,
            values=values,
            names=names,
            title=title,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        return fig

class LineChartVisualizer(Visualizer):
    """Creates line charts"""
    
    def create(self, data, x, y, title, markers=True):
        fig = px.line(
            data,
            x=x,
            y=y,
            title=title,
            markers=markers,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400, hovermode='x unified')
        return fig

class HeatmapVisualizer(Visualizer):
    """Creates heatmap visualizations"""
    
    def create(self, data, title):
        fig = px.density_heatmap(
            data,
            x='publication_year',
            y='Department',
            title=title,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        return fig



class ResearchAnalytics:
    """Performs analytics calculations"""
    
    @staticmethod
    def get_department_stats(df):
        """Calculate department-wise statistics"""
        stats = df.groupby('Department').agg({
            'publication_title': 'count',
            'citations': 'sum'
        }).reset_index()
        stats.columns = ['Department', 'Publications', 'Total Citations']
        stats = stats.sort_values('Publications', ascending=False)
        return stats
    
    @staticmethod
    def get_faculty_stats(df, department):
        """Calculate faculty-wise statistics"""
        filtered = df[df['Department'] == department]
        stats = filtered.groupby('faculty_name').agg({
            'publication_title': 'count',
            'citations': 'sum'
        }).reset_index()
        stats.columns = ['Faculty', 'Publications', 'Total Citations']
        stats = stats.sort_values('Publications', ascending=False)
        return stats
    
    @staticmethod
    def get_yearly_trend(df, filter_col=None, filter_val=None):
        """Get publication trend over years"""
        filtered = df if filter_col is None else df[df[filter_col] == filter_val]
        # Drop NaN years before grouping
        filtered = filtered.dropna(subset=['publication_year'])
        yearly = filtered.groupby('publication_year').size().reset_index()
        yearly.columns = ['Year', 'Publications']
        return yearly.sort_values('Year')
    
    @staticmethod
    def get_publication_types(df, filter_col=None, filter_val=None):
        """Get publication type distribution"""
        filtered = df if filter_col is None else df[df[filter_col] == filter_val]
        
        # Safety check if column exists (it should due to DataProcessor)
        if 'publication_type' in filtered.columns:
            types = filtered['publication_type'].value_counts().reset_index()
            types.columns = ['Type', 'Count']
            return types
        return pd.DataFrame(columns=['Type', 'Count'])
    
    @staticmethod
    def get_research_domains(df, filter_col=None, filter_val=None):
        """Get research domain distribution"""
        filtered = df if filter_col is None else df[df[filter_col] == filter_val]
        domains = filtered['research_domain'].value_counts().reset_index()
        domains.columns = ['Domain', 'Count']
        return domains



def main():
    # Page configuration
    st.set_page_config(
        page_title="Research Publications Dashboard",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #1f77b4 0%, #bbdefb 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .stMetric {
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'view_level' not in st.session_state:
        st.session_state.view_level = 'overview'
        st.session_state.selected_department = None
        st.session_state.selected_faculty = None
    
    # === DATA LOADING AND PROCESSING ===
    try:
        # UPDATED FILENAME TO SHEET 3
        df = pd.read_csv('SSG DATA - Sheet1 (3).csv')
    except Exception as e:
        st.error(f"Error loading 'SSG DATA - Sheet1 (3).csv': {e}")
        return

    # Apply cleaning (This includes the fix for missing publication_type)
    df = DataProcessor.clean_data(df)
    
    # Apply feature extraction for Research Domain
    df['research_domain'] = df['publication_title'].apply(ResearchDomainExtractor.extract_domain)
    # ==================================
    
    if df.empty:
        st.error("No data available. Please check your CSV file.")
        return
    
    # Header
    st.markdown('<div class="main-header">📚 Research Publications Dashboard</div>', unsafe_allow_html=True)
    
    # Navigation
    render_navigation()
    
    # Render appropriate view
    if st.session_state.view_level == 'overview':
        render_overview(df)
    elif st.session_state.view_level == 'department':
        render_department_view(df)
    elif st.session_state.view_level == 'faculty':
        render_faculty_view(df)


def render_navigation():
    """Render navigation breadcrumb"""
    col1, col2, col3 = st.columns([1, 1, 4])
    
    if st.session_state.view_level != 'overview':
        with col1:
            if st.button("⬅️ Back", use_container_width=True):
                if st.session_state.view_level == 'faculty':
                    st.session_state.view_level = 'department'
                    st.session_state.selected_faculty = None # Clear faculty selection
                else: # We are at 'department' view
                    st.session_state.view_level = 'overview'
                    st.session_state.selected_department = None # Clear dept selection
                st.rerun()
    
    with col3:
        breadcrumb = "🏠 Overview"
        if st.session_state.selected_department:
            breadcrumb += f" → 🏢 {st.session_state.selected_department}"
        if st.session_state.selected_faculty:
            breadcrumb += f" → 👤 {st.session_state.selected_faculty}"
        st.markdown(f"### {breadcrumb}")

def render_overview(df):
    """Render overview dashboard"""
    st.markdown("## 📊 Institution-wide Research Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_year_count = 0
    current_year_citations = 0
    if not df['publication_year'].isna().all():
        max_year = df['publication_year'].max()
        current_year_count = len(df[df['publication_year'] == max_year])
        current_year_citations = int(df[df['publication_year'] == max_year]['citations'].sum())

    with col1:
        st.metric("Total Publications", len(df), delta=f"+{current_year_count} this year")
    
    with col2:
        st.metric("Total Citations", int(df['citations'].sum()), delta=f"+{current_year_citations} this year")
    
    with col3:
        st.metric("Departments", df['Department'].nunique())
    
    with col4:
        st.metric("Faculty Members", df['faculty_name'].nunique())
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    # Department-wise publications
    with col1:
        dept_stats = ResearchAnalytics.get_department_stats(df)
        fig = BarChartVisualizer().create(
            dept_stats.head(10),
            x='Publications',
            y='Department',
            title='Top 10 Departments by Publications',
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Make departments clickable
        st.markdown("### 🔍 Explore Departments")
        selected_dept = st.selectbox(
            "Select a department to view details:",
            options=dept_stats['Department'].tolist(),
            key='dept_selector'
        )
        
        if st.button("View Department Details", use_container_width=True):
            st.session_state.selected_department = selected_dept
            st.session_state.view_level = 'department'
            st.rerun()
    
    # Research domains
    with col2:
        domain_stats = ResearchAnalytics.get_research_domains(df)
        fig = PieChartVisualizer().create(
            domain_stats,
            values='Count',
            names='Domain',
            title='Research Domains Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Yearly trend
    st.markdown("### 📈 Publication Trend Over Years")
    yearly_data = ResearchAnalytics.get_yearly_trend(df)
    fig = LineChartVisualizer().create(
        yearly_data,
        x='Year',
        y='Publications',
        title='Publications Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Publication types
    col1, col2 = st.columns(2)
    
    with col1:
        type_stats = ResearchAnalytics.get_publication_types(df)
        fig = PieChartVisualizer().create(
            type_stats,
            values='Count',
            names='Type',
            title='Publication Types'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top cited publications
        st.markdown("### 🏆 Top Cited Publications")
        top_cited = df.nlargest(10, 'citations')[['publication_title', 'faculty_name', 'citations', 'publication_year']]
        st.dataframe(top_cited, use_container_width=True, hide_index=True)

def render_department_view(df):
    """Render department-level dashboard"""
    dept = st.session_state.selected_department
    dept_df = df[df['Department'] == dept].copy()
    
    st.markdown(f"## 🏢 {dept} - Department Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Publications", len(dept_df))
    
    with col2:
        st.metric("Total Citations", int(dept_df['citations'].sum()))
    
    with col3:
        st.metric("Faculty Members", dept_df['faculty_name'].nunique())
    
    with col4:
        avg_citations = 0.0
        if len(dept_df) > 0:
            avg_citations = dept_df['citations'].mean()
        st.metric("Avg Citations/Paper", f"{avg_citations:.1f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    # Faculty-wise publications
    with col1:
        faculty_stats = ResearchAnalytics.get_faculty_stats(df, dept)
        fig = BarChartVisualizer().create(
            faculty_stats.head(15),
            x='Publications',
            y='Faculty',
            title='Faculty Publications',
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Make faculty clickable
        st.markdown("### 👥 Explore Faculty")
        selected_faculty = st.selectbox(
            "Select a faculty member:",
            options=faculty_stats['Faculty'].tolist(),
            key='faculty_selector'
        )
        
        if st.button("View Faculty Profile", use_container_width=True):
            st.session_state.selected_faculty = selected_faculty
            st.session_state.view_level = 'faculty'
            st.rerun()
    
    # Research domains in department
    with col2:
        domain_stats = ResearchAnalytics.get_research_domains(dept_df)
        fig = PieChartVisualizer().create(
            domain_stats,
            values='Count',
            names='Domain',
            title='Research Focus Areas'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Yearly trend for department
    st.markdown("### 📈 Department Publication Trend")
    yearly_data = ResearchAnalytics.get_yearly_trend(df, 'Department', dept)
    fig = LineChartVisualizer().create(
        yearly_data,
        x='Year',
        y='Publications',
        title=f'{dept} - Publications Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Publication types in department
    col1, col2 = st.columns(2)
    
    with col1:
        type_stats = ResearchAnalytics.get_publication_types(dept_df)
        fig = BarChartVisualizer().create(
            type_stats,
            x='Type',
            y='Count',
            title='Publication Types in Department',
            color='Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top publications
        st.markdown("### 🏆 Top Publications")
        top_pubs = dept_df.nlargest(10, 'citations')[['publication_title', 'faculty_name', 'citations', 'publication_year']]
        st.dataframe(top_pubs, use_container_width=True, hide_index=True)

def render_faculty_view(df):
    """Render faculty-level dashboard"""
    faculty = st.session_state.selected_faculty
    faculty_df = df[df['faculty_name'] == faculty].copy()
    
    st.markdown(f"## 👤 {faculty} - Research Profile")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Publications", len(faculty_df))
    
    with col2:
        st.metric("Total Citations", int(faculty_df['citations'].sum()))
    
    with col3:
        # Correct H-Index calculation
        sorted_citations = faculty_df['citations'].sort_values(ascending=False).reset_index(drop=True)
        h_index = 0
        for i, count in enumerate(sorted_citations):
            if count >= i + 1:
                h_index = i + 1
            else:
                break
        st.metric("H-Index", h_index)
    
    with col4:
        avg_citations = 0.0
        if len(faculty_df) > 0:
            avg_citations = faculty_df['citations'].mean()
        st.metric("Avg Citations", f"{avg_citations:.1f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    # Research domains
    with col1:
        domain_stats = ResearchAnalytics.get_research_domains(faculty_df)
        fig = PieChartVisualizer().create(
            domain_stats,
            values='Count',
            names='Domain',
            title='Research Expertise'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Publication types
    with col2:
        type_stats = ResearchAnalytics.get_publication_types(faculty_df)
        fig = PieChartVisualizer().create(
            type_stats,
            values='Count',
            names='Type',
            title='Publication Types'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Yearly productivity
    st.markdown("### 📈 Publication Productivity Over Years")
    
    # Ensure no NaN years
    faculty_df_cleaned = faculty_df.dropna(subset=['publication_year'])
    
    yearly_data = faculty_df_cleaned.groupby('publication_year').agg({
        'publication_title': 'count',
        'citations': 'sum'
    }).reset_index()
    yearly_data.columns = ['Year', 'Publications', 'Citations']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=yearly_data['Year'], y=yearly_data['Publications'], name="Publications"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=yearly_data['Year'], y=yearly_data['Citations'], name="Citations", mode='lines+markers'),
        secondary_y=True
    )
    fig.update_layout(title="Publications and Citations Over Time", height=400)
    fig.update_yaxes(title_text="<b>Publications</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Citations</b>", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Publications table
    st.markdown("### 📄 All Publications")
    
    # Only select journal_name if it exists, keeping the code robust
    cols_to_show = ['publication_title', 'publication_year', 'publication_type', 'citations']
    if 'journal_name' in faculty_df.columns:
        cols_to_show.append('journal_name')
        
    pub_table = faculty_df[cols_to_show].sort_values('citations', ascending=False)
    st.dataframe(pub_table, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()