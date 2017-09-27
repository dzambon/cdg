# cdg: Change Detection Test on a Stream of Graphs #

Author: *Daniele Zambon*, 
Affiliation: *Università della Svizzera italiana* 
eMail: `daniele.zambon@usi.ch` 
Last Update: *17/09/2017* 

## Warning ##

The Delaunay graph dataset adopt an attribute format as comma-separated double values which is not supported by `GraphMatchingToolkit` [1] by default.
You have three options:

1. (Lazy solution) Don't used the Delaunay dataset; 
2. Edit the `database.Delaunay` class to export `.gxl` file in a format compliant to the `database.Letter` dataset.
3. (Recommended) Edit the original `GraphMatchingToolkit` code as follows: extend the `src/CostFunction.java` file of the `GraphMatchingToolkit` with the following function
```
static public double[] dzParseCSVDouble(String str){
    String delims = "[\\[,\\]]";
    String[] tokens = str.split(delims);
    int dim = tokens.length-1;
    double[] x;
    x = new double[dim];
    for (int i=0; i<dim; i++){
        x[i] = Double.parseDouble(tokens[i+1]);
    }
    return x;
}
```
and the following code  
```
[...]
if (this.nodeCostTypes[i].equals("csvDouble")) {
    double[] u_x= dzParseCSVDouble(u
            .getValue(this.nodeAttributes[i]));
    double[] v_x= dzParseCSVDouble(v
            .getValue(this.nodeAttributes[i]));
    double sum = 0;
    for (int s = 0; s < u_x.length; s++) {
        sum += Math.pow((u_x[s] - v_x[s]), 2.);   
    }
    if (this.multiplyNodeCosts == 1) {
        cost *= sum * this.nodeAttrImportance[i];
    } else {
        cost += sum * this.nodeAttrImportance[i];
    }
}
[...]
```
to be added in both the functions
`public double getCost(Node u, Node v)` and `public double getCost(Edge u, Edge v)` in `src/CostFunction.java`. Feel free to ask my support, if necessary.



## References ##

[1] Riesen, Kaspar, Sandro Emmenegger, and Horst Bunke. "A novel software toolkit for graph edit distance computation." International Workshop on Graph-Based Representations in Pattern Recognition. Springer, Berlin, Heidelberg, 2013. 



