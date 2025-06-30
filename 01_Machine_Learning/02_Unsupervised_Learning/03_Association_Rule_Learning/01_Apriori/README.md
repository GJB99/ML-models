# Apriori Algorithm

The Apriori algorithm is used for mining frequent itemsets and devising association rules from a transactional database. The key idea is to use the "Apriori property," which states that all non-empty subsets of a frequent itemset must also be frequent. It systematically identifies frequent itemsets and applies minimum support and confidence thresholds to filter for meaningful rules[34, 35].

### How it works:

1.  **Set a minimum support threshold.**
2.  **Find all frequent itemsets with 1 item.**
3.  **Iteratively generate candidate itemsets of length `k` from frequent itemsets of length `k-1`.** This is the "join" step.
4.  **Prune the candidate itemsets by checking if all their subsets are frequent.** This leverages the Apriori property.
5.  **Repeat until no more frequent itemsets can be found.**
6.  **Generate association rules from the frequent itemsets.** This is done by calculating the confidence for all possible rules and keeping the ones that meet a minimum confidence threshold. 