import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import pandas as pd
from itertools import combinations

# Function to load data (from csv) with dynamic number of rows
def LoadData(filename, percentage):
    DataFrame = pd.read_csv(filename)
    numb_rows = int(len(DataFrame) * (percentage / 100))  # Calculate the number of rows based on percentage
    DataFrame = DataFrame.head(numb_rows)
    Get_transactions = DataFrame.groupby('TransactionNo')['Items'].apply(set).tolist()
    return Get_transactions

# Calculation of support count of an item set in transactions
def SupportCount(AllTransactions, item_set):
    count = 0
    for transaction in AllTransactions:
        unique = set(transaction)
        if(item_set.issubset(unique)):
            count += 1
    return count

# generated candidate item sets have the desired level
def GET_Candidate(FrequentItemSet, level):
    candidates = set()
    for pointer_itemSet1 in FrequentItemSet:
        for pointerItemSet2 in FrequentItemSet:
            if len(pointer_itemSet1.union(pointerItemSet2)) == level:
                candidates.add(pointer_itemSet1.union(pointerItemSet2))
    return candidates


# Apriori Algorithm
def Apiriori_Algo(min_support, Alltransactions):
    item_counts = dict()
    FrequentItemSet = dict()

    for transaction in Alltransactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    # Finding Frequent Item Sets of Level 1:
    FrequentItemSet[1] = {frozenset([item]): count for item, count in item_counts.items() if count >= min_support}

    k = 2
    while FrequentItemSet.get(k-1):
        candidates = GET_Candidate(FrequentItemSet[k-1], k)

        FrequentItemSet[k] = dict()
        for candidate in candidates:
            support = SupportCount(Alltransactions, candidate)
            if support >= min_support:
                FrequentItemSet[k][frozenset(candidate)] = support
        k += 1
    return FrequentItemSet

# Function to generate association rules from frequent item sets


def AssociationRuleCalculation(frequent_item_sets, min_confidence):
    AssociationRuleSet = []
    max_level = max(frequent_item_sets.keys())

    for item_set, support in frequent_item_sets[max_level-1].items():
        for i in range(1, max_level-1):
            for LeftHandSizde in combinations(item_set, i):
                LeftHandSizde = frozenset(LeftHandSizde)
                RightHandSide = item_set - LeftHandSizde
                confidence = support / frequent_item_sets[i].get(LeftHandSizde, 0)
                if confidence >= min_confidence:
                    AssociationRuleSet.append((LeftHandSizde, RightHandSide, confidence))

    return AssociationRuleSet

# Main function
def main():
    def browse_file():
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            entry_filename.delete(0, tk.END)
            entry_filename.insert(0, filename)

    def excuteAlgo():
     filename = entry_filename.get()
     min_support = float(entry_min_support.get())
     min_confidence = float(entry_min_confidence.get())
     percentage = float(entry_percentage.get())  # Retrieve the percentage value
     transactions = LoadData(filename, percentage)  # Pass percentage to LoadData function
     FrequentItemSet = Apiriori_Algo(min_support, transactions)

    # Display results in the text widget
     result_text.delete(1.0, tk.END)
     result_text.insert(tk.END, "Frequent Item Sets:\n\n")
     for k, item_sets in FrequentItemSet.items():
        result_text.insert(tk.END, f"Level {k}:\n")
        for item_set, support in item_sets.items():
            result_text.insert(tk.END, f"{set(item_set)}: Support = {support:.2f}\n")

        # Calculate association rules for the current level
        AssociationRuleSet = AssociationRuleCalculation(FrequentItemSet, min_confidence / 100)
        result_text.insert(tk.END, "\n\n")
        for antecedent, RightHandSide, confidence in AssociationRuleSet:
            if len(antecedent) + len(RightHandSide) == k:  # Filter Association rules by level
                if confidence >= min_confidence / 100:  # Display only if confidence meets the threshold
                    result_text.insert(tk.END, f"{set(antecedent)} => {set(RightHandSide)}: Confidence = {confidence:.2f}\n")


    root = tk.Tk()
    root.title("Apriori Algorithm")
    root.geometry("800x600")

    background_color = "#4A90E2"  # Blue
    button_color = "#000000"      # Black

    root.config(bg=background_color)

    style = ttk.Style()
    style.configure("Bold.TButton", font=('Helvetica', 15, 'bold italic'), foreground="black", background=button_color)

    frame = ttk.Frame(root)
    frame.grid(row=0, column=0, padx=10, pady=10)

    # label_filename = ttk.Label(frame, text="FilePath:")
    label_filename = ttk.Label(frame, text="FilePath:", font=('Helvetica', 12, 'bold'))
    label_filename.grid(row=0, column=0, sticky="w")

    entry_filename = ttk.Entry(frame, width=40)
    entry_filename.grid(row=0, column=1, padx=5, pady=5)

    button_browse = ttk.Button(frame, text="Browse", command=browse_file, style="Bold.TButton")
    button_browse.grid(row=0, column=2, padx=5, pady=5)

    # label_min_support = ttk.Label(frame, text="Minimum Support:")
    label_min_support = ttk.Label(frame, text="Minimum Support:", font=('Helvetica', 12, 'bold'))
    label_min_support.grid(row=1, column=0, sticky="w")

    entry_min_support = ttk.Entry(frame, width=10)
    entry_min_support.grid(row=1, column=1, padx=5, pady=5)

    # label_min_confidence = ttk.Label(frame, text="Minimum Confidence in percentage(%):")
    label_min_confidence = ttk.Label(frame, text="Minimum Confidence in percentage(%):", font=('Helvetica', 12, 'bold'))
    label_min_confidence.grid(row=2, column=0, sticky="w")

    entry_min_confidence = ttk.Entry(frame, width=10)
    entry_min_confidence.grid(row=2, column=1, padx=5, pady=5)

    # Adding label and entry widget for percentage of rows
    label_percentage = ttk.Label(frame, text="Percentage of Data:", font=('Helvetica', 12, 'bold'))
    label_percentage.grid(row=3, column=0, sticky="w")

    entry_percentage = ttk.Entry(frame, width=10)
    entry_percentage.grid(row=3, column=1, padx=5, pady=5)

    button_run = ttk.Button(frame, text="Run", command=excuteAlgo, style="Bold.TButton")
    button_run.grid(row=4, column=0, columnspan=3, pady=10)

    result_text = tk.Text(root, height=30, width=100)  # Increase height to 30 and width to 80
    result_text.grid(row=1, column=0, padx=10, pady=10)
    

    root.mainloop()

if __name__ == "__main__":
    main()
