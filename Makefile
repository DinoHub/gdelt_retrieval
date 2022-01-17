install:
	@echo "Installing dependencies..."
	pip install streamlit streamlit_agraph pandas	

start:
	@echo "Setting up streamlit server..."
	streamlit run GraphQueryApp.py

