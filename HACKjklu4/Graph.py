import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_accuracy(csv_file, exercise_name):
    """
    Plots accuracy from a given CSV file and displays it in Streamlit.

    Parameters:
    - csv_file: Path to the CSV file containing accuracy data.
    - exercise_name: Name of the exercise for labeling the graph.
    """
    try:
        # Load data
        df = pd.read_csv(csv_file)

        # Check if the file has data
        if "Accuracy (%)" in df.columns and not df.empty:
            # Create a figure
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(1, len(df) + 1), df["Accuracy (%)"], color='blue', edgecolor='black', alpha=0.7)

            # Labels and title
            ax.set_xlabel("Exercise Session")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{exercise_name} Accuracy Over Sessions")
            ax.set_xticks(range(1, len(df) + 1))
            ax.set_ylim(0, 100)  # Accuracy is in percentage
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Display plot in Streamlit
            st.pyplot(fig)
        else:
            st.warning(f"No data available in {csv_file} to plot.")
    except FileNotFoundError:
        st.error(f"Error: {csv_file} not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Example usage inside Streamlit
def show_graphs():
    st.write("### Biceps Curl Accuracy")
    plot_accuracy("Biceps-final.csv", "Biceps Curl")

    st.write("### Squats Accuracy")
    plot_accuracy("squarts-final.csv", "Squats")

# Run only if script is executed directly
if __name__ == "__main__":
    show_graphs()
