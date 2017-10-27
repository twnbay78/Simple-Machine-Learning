<h3>Summary</h3>

In the new age of Artificial Intelligence, machine learning has become quite popular 
due to how powerful it can be. One field that utilizes machine learning quite frequently
is finance, particularly in the real-estate sector. 

This program will take in training data from a .txt file consisting of house attributes (number
of bathrooms, year the house was built, size of the house, etc...), and minimize the sum 
of squared residuals, or "data fits" the systems of equations in order to find weights 
associated with the data. The weights are then applied to test data, which consists of
solely house attributes in order to estimate the cost of a house. 

<h3>Usage</h3>

Compiling and linking necessary files is made simple using the "make" command in terminal:
<br>
<br>
<pre>$make</pre>
<br>

Runnint the program is accomplished by initializing the following terminal command:
<br>
<br>
<pre>$./learn <em>train_file</em>.txt <em>test_file</em>.txt</pre>
<br>

<h3>Implementation</h3>

Calculating the weights needed to estimate the house cost is done by computing the 
Moore-Penrose "pseudoinverse." the equation is as follows: 
<br>
<em>W = (XTT * X)^−1 * X^T * Y</em>
<br>
<br>
<h5>learn.c</h5>
<br>
<ul>
	<li>

