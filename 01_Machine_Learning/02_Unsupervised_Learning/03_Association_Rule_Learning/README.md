# Association Rules and Market Basket Analysis

Association rule learning identifies patterns and relationships between variables in large datasets, particularly useful for understanding customer purchasing behavior[34][35]. Market basket analysis represents the most common application, discovering which products customers frequently purchase together[34].

### Key Metrics

**Support**: Measures the frequency of item occurrence across all transactions[34][35]. High support indicates items frequently purchased by customers[35].

**Confidence**: Calculates the conditional probability of purchasing item Y given item X was purchased[34][35]. High confidence suggests strong predictive relationships between items[35].

**Lift**: Compares observed co-occurrence frequency with expected frequency under independence assumption[35]. Lift values greater than 1 indicate positive correlation between items[35].

### Algorithmic Approaches

Common algorithms include:
- **Apriori Algorithm**: Uses frequent itemset mining to identify association rules systematically.
- **FP-Growth**: Provides an efficient alternative to Apriori through frequent pattern tree construction. 