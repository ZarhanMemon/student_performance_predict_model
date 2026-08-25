import pandas as pd

# Read the two datasets
d1 = pd.read_csv("data/student-mat.csv", sep=";")
d2 = pd.read_csv("data/student-por.csv", sep=";")

# Columns used for merging
merge_columns = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet"
]

# Merge the Mathematics and Portuguese datasets
d3 = pd.merge(
    d1,
    d2,
    on=merge_columns
)

#combined data
d3.to_csv("data/student-merge.csv", index=1,index_label="id" )


# Number of students after merging
print("Number of students:", len(d3))