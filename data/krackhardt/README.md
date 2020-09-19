Dowloaded on 2020-09-17 from:https://comunelab.fbk.eu/data/Krackhardt-High-Tech_Multiplex_Social.zip

# KRACKHARDT-HIGH-TECH MULTIPLEX NETWORK

###### Last update: 1 July 2014

### Reference and Acknowledgments

This README file accompanies the dataset representing the multiplex social network of the managers of a high-tech company.
If you use this dataset in your work either for analysis or for visualization, you should acknowledge/cite the following papers:

	"Cognitive social structures"
	D. Krackhardt
	Social Networks (1987), 9, 104-134


### Description of the dataset

The multiplex social network consists of 3 kinds of relationships (Advice, Friendship and "Reports to") between managers of a high-tech company.

There are 21 nodes in total, labelled with integer ID between 1 and 21, with 312 connections.
The multiplex is directed and unweighted, stored as edges list in the file
    
    Krackhardt-High-Tech_multiplex.edges

with format

    layerID nodeID nodeID weight

(Note: all weights are set to 1)

The IDs of all layers are stored in 

    Krackhardt-High-Tech_layers.txt

The IDs of nodes can be found in the file

    Krackhardt-High-Tech_nodes.txt

The values for the 5 columns (attributes) are

1. node id
2. age (in years)
3. tenure (length of service or tenure in years)
4. level (in the corporate hierarchy; 1 = CEO, 2 = Vice President, 3 = manager)
5. deparment (coded 1,2,3,4 with the CEO in department 0 ie not in a department)


### License

The KRACKHARDT-HIGH-TECH MULTIPLEX DATASET is provided "as is" and without warranties as to performance or quality or any other warranties whether expressed or implied. 

