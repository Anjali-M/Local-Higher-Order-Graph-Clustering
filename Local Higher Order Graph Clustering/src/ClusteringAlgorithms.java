import java.io.*;
import java.util.*;
import org.apache.commons.math3.*;
import org.apache.commons.math3.linear.Array2DRowFieldMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

public class ClusteringAlgorithms {
	
	//V :Nodes; E:Edges
	static int V, E;
	
	//G is the graph matrix
	static int[][] G;
	
	//e is a unit column vector(used in computation). 
	static int[][] e;
	
	public static void main(String[] args) throws FileNotFoundException
	{
		
		double alpha = 0.98; //Teleportation parameter
		double epsilon = 0.0001; //Tolerance
		int u=1; //seed node
		
		File myObj=new File(".//src//Dataset_M7.txt");
		Scanner input=new Scanner(myObj);
		
		Scanner userip=new Scanner(System.in);
		//Please select the motif in accordance with the graph and seed node to get appropriate results
		System.out.println("Enter the Motif M(1-7) : ");
		int motif=userip.nextInt();
		
		V=input.nextInt();
		E=input.nextInt();
		
		G=new int[V][V];
		
		while(input.hasNext())
		{
			int x=input.nextInt();
			int y=input.nextInt();
			
			G[x][y]=1;
		}
		
		e=new int[V][1];
		for(int i=0;i<V;i++)
			e[i][0]=1;
		
		//1st 
		//Global Motif Clustering..
		long startTime = System.currentTimeMillis();
				
		List<Integer> globalMotifCluster=GlobalMAPPR(G, motif, u,alpha, epsilon );
				
		if(globalMotifCluster==null)
		{	
			System.out.println("Execution Termination...Please change dataset...");
		}
		else
		{	
			System.out.println("The Global motif Cluster for the seed node(u:"+u+"), Motif (M"+motif+") : ");
			for(int i=0;i<globalMotifCluster.size();i++)
				System.out.print(globalMotifCluster.get(i)+" ");
			System.out.println();
			
			long endTime =System.currentTimeMillis();;		
					
			System.out.println("Execution Time for Global Motif"+motif+" : "+((double) (endTime-startTime))+" milli seconds");
		}
		
		//2nd
		//Global Motif Clustering.
		startTime = System.currentTimeMillis();
		
		List<Integer> localMotifCluster=MAPPR(G, motif, u,alpha, epsilon );
		
		if(localMotifCluster==null)
		{	
			System.out.println("Execution Termination...Please change dataset...");
		}
		else
		{
			System.out.println("The local motif Cluster(for the seed node(u :"+u+"), Motif (M"+motif+") : ");
			for(int i=0;i<localMotifCluster.size();i++)
				System.out.print(localMotifCluster.get(i)+" ");
			System.out.println();
					
			long endTime =System.currentTimeMillis();
							
			System.out.println("Execution Time for Local Motif"+motif+" : "+(double) (endTime-startTime)+" milli seconds");	
		}
		

		// 3rd
		//Local-Edge based Algorithm..
		startTime = System.nanoTime();
				
		List<Integer> LocalEdgeCluster=APPR(G,u,alpha, epsilon );
				
		if(LocalEdgeCluster==null)
		{	
			System.out.println("Execution Termination...Please change dataset...");
		}
		else
		{	
			System.out.println("The Local Edge Cluster for the seed node(u:"+u+"), Motif (M"+motif+") : ");
			for(int i=0;i<LocalEdgeCluster.size();i++)
				System.out.print(LocalEdgeCluster.get(i)+" ");
			System.out.println();
			
			long endTime =System.nanoTime();;		
					
			System.out.println("Execution Time for Local Edge Cluster : "+((double) (endTime-startTime))+" nano-seconds : "+((double) (endTime-startTime)/1000000)+" milli-seconds");
		}
		
	}
	
	static List<Integer> MAPPR(int[][] G, int M, int u, double alpha, double epsilon)
	{
		int[][] W=new int[V][V];
		int[][] U=new int[V][V];
		int[][] B=new int[V][V];
		
		double[][] PPRVector;
		double[][] Dw=new double[V][V];
		double[][] DwIxP=new double[V][1];
		int Ew=0;
		
		//Construct Weighted Graph.....
		B=hMatrixMultiplication(G,matrixTranspose(G));
		U=matrixSubtraction(G,B);
		
		switch(M)
		{
			case 1: W=motifMatrix_M1(G,B,U);
					break;
			case 2: W=motifMatrix_M2(G,B,U);
					break;
			case 3: W=motifMatrix_M3(G,B,U);
					break;
			case 4: W=motifMatrix_M4(G,B,U);
					break;
			case 5: W=motifMatrix_M5(G,B,U);
					break;
			case 6: W=motifMatrix_M6(G,B,U);
					break;
			case 7: W=motifMatrix_M7(G,B,U);
					break;			
		}
	
		for(int i=0;i<V;i++)
		{
			for(int j=0;j<V;j++)
			{
				if(W[i][j]!=0)
					Ew++;
			}
		}
		//Since, W is an undirected matrix, each edge is counted twice
		Ew=Ew/2;
		
		PPRVector = WeightedAPPR(W,V,Ew,u,alpha, epsilon);
		
		//Constructing the diagonal degree matrix
		for(int i=0;i<V;i++)
		{
			for(int j=0;j<V;j++)
				Dw[i][i]+=W[i][j];
		}
		
		//When the motif (or) seed is not chosen properly, We can't get the inverse matrix since det(Dw) becomes 0 
		try 
		{
			double[][] DwInverse=inverse(Dw);
			DwIxP=mnMatrixMul(DwInverse, PPRVector);
		}
		catch(Exception e)
		{
			System.out.println("\nPLEASE NOTE : The graph data being used is not suitable for this Motif...\n");
			System.out.println("	(Not having enough motifs of the specified type in the graph data maybe the cause of this issue.)\n");
			System.out.println("	(The Determinant of the Weighted Motif Degree Matrix is 0 (Singular Matrix),so inverse can't be found.)\n");
			System.out.println("	Please try a different graph for the chosen motif...\n");
			System.out.println("	e.g., Use Dataset_M7.txt for M7, Dataset_M2 for M2 (provided in the src)\n");
			return null;
		}
		
		int[] eigenVector=new int[V];
		for(int i=0;i<V;i++)
			eigenVector[i]=i;
		
		sort(eigenVector, DwIxP, 0,V-1);
		
		//CLUSTERING (Finding motifCuts, motifVolumes for finding MotifConductance of each cluster
		//Find MotifConductance for each cluster, stop when minMotifConductance > 1.2*(prevMinMotifConductance).
		//It means that the local minimum -k for optimal cluster size (Sk) is found
		//Return the obtained cluster at this point
		List<Integer> S=new LinkedList<Integer>();
		
		int k=3;
		int localminimum=3;
		int cutMotifCount=0;
		int cutMotifVolume=k;
		double motifConductance=0;
		double minMotifConducatance=0;
		for(int i=0;i<k;i++)
			S.add(eigenVector[i]);
		
		while(true)
		{
			cutMotifCount=0;
			motifConductance=0;
			
			if(k!=3)
			{
				List<Integer> visited=new LinkedList<Integer>();
				int newNode=eigenVector[k-1];
				
				for(int i=0;i<k && !visited.contains(eigenVector[i]);i++)
				{
					visited.add(eigenVector[i]);
					
					if(W[newNode][eigenVector[i]]!=0)
					{
						boolean newMotif=false;
						for(int j=i+1;j<k && !visited.contains(eigenVector[j]);j++)
						{
								visited.add(eigenVector[j]);
							
								if(W[newNode][eigenVector[j]]!=0 && W[eigenVector[j]][eigenVector[i]]!=0)
								{	newMotif=true;
									cutMotifVolume++;
								}
							
						}
						if(newMotif==false)
						{
							for(int j=k;j<V && !visited.contains(eigenVector[j]) ;j++)
							{
								visited.add(eigenVector[j]);
								if(W[newNode][eigenVector[j]]!=0)
								{
									if(W[eigenVector[i]][eigenVector[j]]!=0 && W[eigenVector[i]][newNode]!=0)
										cutMotifVolume++;
									if(j+1<V && W[newNode][eigenVector[j+1]]!=0)
									{
										for(int next=j+1;next<V && !visited.contains(eigenVector[next]);next++)
										{
											visited.add(eigenVector[next]);
											if(W[newNode][eigenVector[j]]!=0 && W[eigenVector[j]][next]!=0)
												cutMotifVolume++;
										}
									}
									
									//cutMotifVolume+=2;
									//System.out.println("Entering +2 , k="+k+"newNode="+newNode+"(i,j)"+i+j);
								}
								
							}
						}
					}
				}
			}
			List<Integer> cutMotifSet=new LinkedList<Integer>();
			
			for(int i=0;i<k;i++)
			{	
				for(int j=0;j<V;j++)
				{
					if(W[i][j]!=0)
					{
						if(!S.contains(j) && !cutMotifSet.contains(j))
						{
							if(cutMotifSet.size()==0)
							{
								cutMotifCount++;
								if(k==3)
									cutMotifVolume++;
							}
							else 
							{
								int connected=-1;
								
								for(int q=0;q<cutMotifSet.size();q++)
								{
									int x=cutMotifSet.get(q);
									
									if(W[j][x]!=0)
									{
										connected=x;
										if(W[S.get(i)][j]!=0 && W[S.get(i)][x]!=0)
											cutMotifCount++;
									}
								}
								if(connected==-1)
								{
									cutMotifCount++;
									if( k==3)
										cutMotifVolume++;
								}	
							}
							
							cutMotifSet.add(j);
						}
						else if(!S.contains(j) && cutMotifSet.contains(j))
						{
							int connected=-1;	
							for(int q=0;q<cutMotifSet.size() && cutMotifSet.get(q)!=j;q++)
							{
								if(W[j][cutMotifSet.get(q)]!=0)
									connected=1;
							}		
							if(connected==-1 && k==3)
								cutMotifVolume++;		
						}
						
					}
				}	
			}
		
			
			motifConductance=(double) cutMotifCount/cutMotifVolume;
	
			if(k==3)
				minMotifConducatance=motifConductance;
			
			else if(minMotifConducatance > motifConductance)
			{
				minMotifConducatance=motifConductance;
				localminimum=k;
			}
				
			if(motifConductance > 1.2*minMotifConducatance)
			{
				S.remove(k-1);
				return S;
			}
					
			if(k < eigenVector.length)
			{
				S.add(eigenVector[k]);
				k++;
			}
			else
				break;	
		}
		
		return S;
	}
	
	//MAPPR method for Global Motif CLustering
	static List<Integer> GlobalMAPPR(int[][] G, int M, int u, double alpha, double epsilon)
	{
		int[][] W=new int[V][V];
		int[][] U=new int[V][V];
		int[][] B=new int[V][V];
		
		double[][] PPRVector;
		double[][] Dw=new double[V][V];
		double[][] DwIxP=new double[V][1];
		int Ew=0;
		
		//Construct Weighted Graph.....
		B=hMatrixMultiplication(G,matrixTranspose(G));
		U=matrixSubtraction(G,B);
		
		switch(M)
		{
			case 1: W=motifMatrix_M1(G,B,U);
					break;
			case 2: W=motifMatrix_M2(G,B,U);
					break;
			case 3: W=motifMatrix_M3(G,B,U);
					break;
			case 4: W=motifMatrix_M4(G,B,U);
					break;
			case 5: W=motifMatrix_M5(G,B,U);
					break;
			case 6: W=motifMatrix_M6(G,B,U);
					break;
			case 7: W=motifMatrix_M7(G,B,U);
					break;			
		}
		
		for(int i=0;i<V;i++)
		{
			for(int j=0;j<V;j++)
			{
				if(W[i][j]!=0)
					Ew++;
			}
		}
		//Since, W is an undirected matrix, each edge is counted twice
		Ew=Ew/2;
		
		PPRVector = WeightedAPPR(W,V,Ew,u,alpha, epsilon);
		
		for(int i=0;i<V;i++)
		{
			for(int j=0;j<V;j++)
				Dw[i][i]+=W[i][j];
		}
		
		//When the motif (or) seed is not chosen properly, We can't get the inverse matrix since det(Dw) becomes 0
		try 
		{
			double[][] DwInverse=inverse(Dw);
			DwIxP=mnMatrixMul(DwInverse, PPRVector);
		}
		catch(Exception e)
		{
			System.out.println("\nPLEASE NOTE : The graph data being used is not suitable for this Motif...\n");
			System.out.println("	(Not having enough motifs of the specified type in the graph data maybe the cause of this issue.)\n");
			System.out.println("	(The Determinant of the Weighted Motif Degree Matrix is 0 (Singular Matrix),so inverse can't be found.)\n");
			System.out.println("	Please try a different graph for the chosen motif...\n");
			System.out.println("	e.g., Use Dataset_M7.txt for M7, Dataset_M2 for M2 (provided in the src)\n");
			return null;
		}
		
		int[] eigenVector=new int[V];
		for(int i=0;i<V;i++)
			eigenVector[i]=i;
		
		sort(eigenVector, DwIxP, 0,V-1);
		
		//CLUSTERING (Finding motifCuts, motifVolumes for finding MotifConductance of each cluster
		//FInd MotifConductance for all clusters, then choose the low MotifConductance cluster as output
		List<Integer> S=new LinkedList<Integer>();
			
		int k=3;
		int koptimal=3;
		int cutMotifCount=0;
		int cutMotifVolume=k;
		double motifConductance=0;
		double minMotifConducatance=0;
		for(int i=0;i<k;i++)
			S.add(eigenVector[i]);
		
		while(true)
		{
			cutMotifCount=0;
			motifConductance=0;
			
			if(k!=3)
			{
				List<Integer> visited=new LinkedList<Integer>();
				int newNode=eigenVector[k-1];
				
				for(int i=0;i<k && !visited.contains(eigenVector[i]);i++)
				{
					visited.add(eigenVector[i]);
					
					if(W[newNode][eigenVector[i]]!=0)
					{
						boolean newMotif=false;
						for(int j=i+1;j<k && !visited.contains(eigenVector[j]);j++)
						{
								visited.add(eigenVector[j]);
								
								if(W[newNode][eigenVector[j]]!=0 && W[eigenVector[j]][eigenVector[i]]!=0)
								{	newMotif=true;
									cutMotifVolume++;
								}
							
						}
						if(newMotif==false)
						{
							for(int j=k;j<V && !visited.contains(eigenVector[j]) ;j++)
							{
								visited.add(eigenVector[j]);
								if(W[newNode][eigenVector[j]]!=0)
								{
									if(W[eigenVector[i]][eigenVector[j]]!=0 && W[eigenVector[i]][newNode]!=0)
										cutMotifVolume++;
									if(j+1<V && W[newNode][eigenVector[j+1]]!=0)
									{
										for(int next=j+1;next<V && !visited.contains(eigenVector[next]);next++)
										{
											visited.add(eigenVector[next]);
											if(W[newNode][eigenVector[j]]!=0 && W[eigenVector[j]][next]!=0)
												cutMotifVolume++;
										}
									}
									
									//cutMotifVolume+=2;
									//System.out.println("Entering +2 , k="+k+"newNode="+newNode+"(i,j)"+i+j);
								}
								
							}
						}
					}
				}
			}
			List<Integer> cutMotifSet=new LinkedList<Integer>();
			
			for(int i=0;i<k;i++)
			{	
				for(int j=0;j<V;j++)
				{
					if(W[i][j]!=0)
					{
						if(!S.contains(j) && !cutMotifSet.contains(j))
						{
							if(cutMotifSet.size()==0)
							{
								cutMotifCount++;
								if(k==3)
									cutMotifVolume++;
							}
							else 
							{
								int connected=-1;
								
								for(int q=0;q<cutMotifSet.size();q++)
								{
									int x=cutMotifSet.get(q);
									
									if(W[j][x]!=0)
									{
										connected=x;
										if(W[S.get(i)][j]!=0 && W[S.get(i)][x]!=0)
											cutMotifCount++;
									}
								}
								if(connected==-1)
								{
									cutMotifCount++;
									if( k==3)
										cutMotifVolume++;
								}	
							}
							
							cutMotifSet.add(j);
						}
						else if(!S.contains(j) && cutMotifSet.contains(j))
						{
							int connected=-1;	
							for(int q=0;q<cutMotifSet.size() && cutMotifSet.get(q)!=j;q++)
							{
								if(W[j][cutMotifSet.get(q)]!=0)
									connected=1;
							}		
							if(connected==-1 && k==3)
								cutMotifVolume++;		
						}
						
					}
				}	
			}
		
			
			motifConductance=(double) cutMotifCount/cutMotifVolume;
				
			//System.out.println("k -> "+k+" : mc, mv,motifCond: "+cutMotifCount+","+cutMotifVolume+","+motifConductance);
				
			if(k==3)
				minMotifConducatance=motifConductance;
			
			else if(minMotifConducatance > motifConductance)
			{
				if(motifConductance > 0)
				{
					minMotifConducatance=motifConductance;
					koptimal=k;
				}
				
			}
	
			if(k < eigenVector.length)
			{
				S.add(eigenVector[k]);
				k++;
			}
			else
				break;	
		}
		
		for(int i=koptimal;i<V;i++)
			S.remove(i);
		
		return S;
	}
	
	//Method for Local-Edge Based Clustering Approach
	static List<Integer> APPR(int[][] G, int u, double alpha, double epsilon)
	{
		double[][] PPRVector;
	
		PPRVector = EgdeAPPR(G,V,E,u,alpha, epsilon);
		
		int[] eigenVector=new int[V];
		for(int i=0;i<V;i++)
			eigenVector[i]=i;
		
		sort(eigenVector, PPRVector, 0,V-1);
		
		//CLUSTERING
		//Stop when the k-optimal is found, based on minEdgeConductance
		List<Integer> S=new LinkedList<Integer>();
		
		int k=0;
		int localMinimum=0;
		int edgeCut=0;
		double conductance=0;
		double minConductance=Integer.MAX_VALUE;
		int totalDegree=0;
		
		List<Integer> visited=new LinkedList<Integer>();
		List<Integer> clusterSet=new LinkedList<Integer>();
		
		for(int i=0;i<V;i++)
		{
			clusterSet.add(eigenVector[i]);
			visited.add(i);
			
			edgeCut=0;
		
			for(int j=0;j<V;j++)
			{
				totalDegree+=G[i][j];
				if(G[i][j]!=0 && visited.contains(j))
					break;
				else
				{
					edgeCut++;
				}
			}
			
			conductance=edgeCut/totalDegree;
			
			if(minConductance > conductance)
			{
				localMinimum = clusterSet.size();
				minConductance = conductance;
			}
			else if(conductance > 1.2*minConductance)
			{
				return clusterSet;
			}
		}
		
		return clusterSet;
	}
	
	//Returns the PageRank Vector for the Local-Edge based CLustering
	static double[][] EgdeAPPR(int[][] G,int V, int E,int u, double alpha, double epsilon )
	{
		double[][] vectorPPR=new double[V][1];
		Queue<Integer> apprQueue=new LinkedList<Integer>();
		
		int pushedNodes=0;
		
		//adding seed node to the queue
		apprQueue.add(u);
		int nodeId=u;
		
		for(int i=0;i<V;i++)
			vectorPPR[i][0]=0;
		
		double[] r=new double[V];
		for(int i=0;i<V;i++)
		{
			if(i==u)
				r[i]=1;
			else
				r[i]=0;
		}
		
		double[] dw=new double[V];
		for(int i=0;i<V;i++)
		{
			for(int j=0;j<V;j++)
				dw[i]+=G[i][j];
		}
		
		while(!apprQueue.isEmpty())
		{
			pushedNodes+=1;
			
			nodeId=apprQueue.poll();
			
			if((r[nodeId]/dw[nodeId]) >= epsilon)
			{

				vectorPPR[nodeId][0]=vectorPPR[nodeId][0]+(alpha* r[nodeId]);
				r[nodeId]=(1-alpha)*(r[nodeId]/2.0);
				
				for(int j=0;j<V;j++)
				{
					if(G[nodeId][j] != 0)
					{
						r[j]=(double) (r[j]+(1-alpha)*r[nodeId]/(2*dw[nodeId]));
						apprQueue.add(j);
					}
				}
				
			}
			
		}
	
		return vectorPPR;
		
	}
	
	//Returns the PageRank Vector for the Motif based CLustering techniques.
	static double[][] WeightedAPPR(int[][] W,int Vw, int Ew,int u, double alpha, double epsilon )
	{
		double[][] vectorPPR=new double[Vw][1];
		Queue<Integer> apprQueue=new LinkedList<Integer>();
		
		int pushedNodes=0;
		
		//adding seed node to the queue
		apprQueue.add(u);
		int nodeId=u;
		
		for(int i=0;i<Vw;i++)
			vectorPPR[i][0]=0;
		
		double[] r=new double[Vw];
		for(int i=0;i<Vw;i++)
		{
			if(i==u)
				r[i]=1;
			else
				r[i]=0;
		}
		
		double[] dw=new double[Vw];
		for(int i=0;i<Vw;i++)
		{
			for(int j=0;j<Vw;j++)
				dw[i]+=W[i][j];
		}
		
		while(!apprQueue.isEmpty())
		{
			pushedNodes+=1;
			
			nodeId=apprQueue.poll();
			
			if((r[nodeId]/dw[nodeId]) >= epsilon)
			{
				double temp=r[nodeId]-((double)(epsilon/2.0))*dw[nodeId];
				vectorPPR[nodeId][0]=(double) (vectorPPR[nodeId][0]+(1-alpha)*temp);
				r[nodeId]=(double) ((epsilon/2.0)*dw[nodeId]);
					
				for(int j=0;j<Vw;j++)
				{
					if(W[nodeId][j] != 0)
					{
						r[j]=(double) (r[j]+(W[nodeId][j]/dw[nodeId]) * alpha * temp);
						apprQueue.add(j);
					}
				}
				
			}
			
		}
	
		return vectorPPR;
		
	}
	
	//These are the Motif-generating functions
	static int[][] motifMatrix_M1(int[][] G,int[][] B,int[][] U)
	{
		int[][] C= hMatrixMultiplication(matrixMultiplication(U,U),matrixTranspose(U));
		
		return matrixAddition(C,matrixTranspose(C));
	}
	
	static int[][] motifMatrix_M2(int[][] G,int[][] B,int[][] U)
	{
		int[][] temp1=new int[V][V];
		int[][] temp2=new int[V][V];
		int[][] temp3=new int[V][V];
		
		temp1=hMatrixMultiplication(matrixMultiplication(B,U),matrixTranspose(U));
		temp2=hMatrixMultiplication(matrixMultiplication(U,B),matrixTranspose(U));
		temp3=hMatrixMultiplication(matrixMultiplication(U,U),B);
		
		int[][] C=matrixAddition(temp1,matrixAddition(temp2,temp3));
		
		return matrixAddition(C,matrixTranspose(C));
		
	}
	
	static int[][] motifMatrix_M3(int[][] G,int[][] B,int[][] U)
	{
		int[][] temp1=new int[V][V];
		int[][] temp2=new int[V][V];
		int[][] temp3=new int[V][V];
		
		temp1=hMatrixMultiplication(matrixMultiplication(B,B),U);
		temp2=hMatrixMultiplication(matrixMultiplication(B,U),B);
		temp3=hMatrixMultiplication(matrixMultiplication(U,B),B);
		
		int[][] C=matrixAddition(temp1,matrixAddition(temp2,temp3));
		
		return matrixAddition(C,matrixTranspose(C));
		
	}
	
	static int[][] motifMatrix_M4(int[][] G,int[][] B,int[][] U)
	{
		return hMatrixMultiplication(matrixMultiplication(B,B),matrixTranspose(B));
		
	}
	
	static int[][] motifMatrix_M5(int[][] G,int[][] B,int[][] U)
	{
		int[][] temp1=new int[V][V];
		int[][] temp2=new int[V][V];
		int[][] temp3=new int[V][V];
		
		temp1=hMatrixMultiplication(matrixMultiplication(U,U),U);
		temp2=hMatrixMultiplication(matrixMultiplication(U,matrixTranspose(U)),U);
		temp3=hMatrixMultiplication(matrixMultiplication(matrixTranspose(U),U),U);
		
		int[][] C=matrixAddition(temp1,matrixAddition(temp2,temp3));
		
		return matrixAddition(C,matrixTranspose(C));
		
	}
	
	static int[][] motifMatrix_M6(int[][] G,int[][] B,int[][] U)
	{
		int[][] temp1=new int[V][V];
		int[][] temp2=new int[V][V];
		int[][] temp3=new int[V][V];
		
		temp1=hMatrixMultiplication(matrixMultiplication(U,B),U);
		temp2=hMatrixMultiplication(matrixMultiplication(B,matrixTranspose(U)),matrixTranspose(U));
		temp3=hMatrixMultiplication(matrixMultiplication(matrixTranspose(U),U),B);
		
		return matrixAddition(temp1,matrixAddition(temp2,temp3));
	}
	
	static int[][] motifMatrix_M7(int[][] G,int[][] B,int[][] U)
	{	
		int[][] temp1=new int[V][V];
		int[][] temp2=new int[V][V];
		int[][] temp3=new int[V][V];
		
		temp1=hMatrixMultiplication(matrixMultiplication(matrixTranspose(U),B),matrixTranspose(U));
		temp2=hMatrixMultiplication(matrixMultiplication(B,U),U);
		temp3=hMatrixMultiplication(matrixMultiplication(U,matrixTranspose(U)),B);
		
		return matrixAddition(temp1,matrixAddition(temp2,temp3));
	}
	
	//Remaining are matrix-functions for computing intermediaryresults.
	//The main function that implements QuickSort() 
	static void sort(int ev[],double dwIpv[][], int low, int high) 
    { 
        if (low < high) 
        { 
            
            int pi = partition(ev,dwIpv,low, high); 
  
            // Recursively sort elements before partition and after partition 
            sort(ev,dwIpv, low, pi-1); 
            sort(ev,dwIpv, pi+1, high); 
        } 
    } 

	static int partition(int ev[],double dwIpv[][], int low, int high) 
    { 
        double pivot = dwIpv[high][0];  
        int i = (low-1); // index of smaller element 
        for (int j=low; j<high; j++) 
        { 
            // If current element is smaller than the pivot 
            if (dwIpv[j][0] > pivot) 
            { 
                i++; 
  
                // swap in eigen vector
                int temp = ev[i]; 
                ev[i] = ev[j]; 
                ev[j] = temp;
                
                //Simultaneously swap in StartTime Array  
                double temp1 = dwIpv[i][0]; 
                dwIpv[i][0] = dwIpv[j][0]; 
                dwIpv[j][0] = temp1;
            } 
        } 
  
        // swap end[i+1] and end[high] (or pivot) 
        int temp = ev[i+1]; 
        ev[i+1] = ev[high]; 
        ev[high] = temp; 
        
      //Simultaneously swap in StartTime Array  
        double temp1 = dwIpv[i+1][0]; 
        dwIpv[i+1][0] = dwIpv[high][0]; 
        dwIpv[high][0] = temp1;
  
        return i+1; 
    } 
	static double[][] mnMatrixMul(double[][] dwInverse, double[][] PPRVector) {
		
		// TODO Auto-generated method stub
		int r1=dwInverse.length;
		int c1=dwInverse[0].length;
		int r2=PPRVector.length;
		int c2=PPRVector[0].length;
		
		if(r2 != c1)
		{
			System.out.println("DwInverse x PPRVector multiplcation is not possible");
			System.out.println("r1,c1,r2,c2 : "+r1+","+c1+","+r2+","+c2);
			return null;
		}
		
		double[][] product=new double[r1][c1];
		
		for(int i=0;i<r1;i++)
		{
			for(int j=0;j<c2;j++)
			{
				for(int k=0;k<r2;k++)
				{
					product[i][j]+=dwInverse[i][k]*PPRVector[k][j];
				}
			}
		}
		return product;
	}
  
	// Function to calculate and store inverse, returns false if 
	// matrix is singular 
	//Jar files is used for efficiency
	static double[][] inverse(double[][] A) 
	{ 
	    RealMatrix matrix=new Array2DRowRealMatrix(A);
	               
	    RealMatrix solver=new LUDecomposition(matrix).getSolver().getInverse();
	    
	    double[][] inverse=new double[V][V];
	    
	    for (int i = 0; i < V; i++) 
	        for (int j = 0; j < V; j++) 
	            inverse[i][j] = (double) solver.getEntry(i, j);
	    
	    return inverse; 
	} 
	static int[][] matrixAddition(int[][] a, int[][] b)
	{
		int c[][]=new int[V][V];  //since adjacency matrix are VxV  
	    
		// addition of 2 matrices    
		for(int i=0;i<V;i++)
		{    
			for(int j=0;j<V;j++)    
				c[i][j]=a[i][j]+b[i][j];    //use - for subtraction  
	    	}

		return c;
	}  
	
	static int[][] matrixSubtraction(int[][] a, int[][] b)
	{
		int c[][]=new int[V][V];  //since adjacency matrix are VxV  
	    
		//subtraction of 2 matrices    
		for(int i=0;i<V;i++)
		{    
			for(int j=0;j<V;j++)    
				c[i][j]=a[i][j]-b[i][j];    //use - for subtraction  
	    	}

		return c;
	}  
	
	static double[][] matrixDoubleIntSubtraction(int[][] a, double[][] b)
	{
		double c[][]=new double[V][V];  //since adjacency matrix are VxV  
	    
		//subtraction of 2 matrices    
		for(int i=0;i<V;i++)
		{    
			for(int j=0;j<V;j++)    
				c[i][j]=a[i][j]-b[i][j];    //use - for subtraction  
	    	}

		return c;
	}  

	static int[][] matrixMultiplication(int[][] A, int[][] B)
	{
		// Matrix to store the result 
	    // since adjacency matrices are VxV, the product matrix will also be VxV
		int C[][]=new int[V][V];    
	  
	        int i, j, k; 
	  
	        // Multiply the two matrices 
	        for (i = 0; i < V; i++) { 
	            for (j = 0; j < V; j++) { 
	                for (k = 0; k < V; k++) 
	                    C[i][j] += A[i][k] * B[k][j]; 
	            } 
	        } 

		return C;
	} 
	
	static double[][] matrixDoubleMultiplication(double[][] A, int[][] B)
	{
		// Matrix to store the result 
	    // since adjacency matrices are VxV, the product matrix will also be VxV
		double C[][]=new double[V][V];    
	  
	        int i, j, k; 
	  
	        // Multiply the two matrices 
	        for (i = 0; i < V; i++) { 
	            for (j = 0; j < V; j++) { 
	                for (k = 0; k < V; k++) 
	                    C[i][j] += A[i][k] * B[k][j]; 
	            } 
	        } 

		return C;
	} 
	
	static double[][] matrixDoubleDoubleMultiplication(double[][] A, double[][] B)
	{
		// Matrix to store the result 
	    // since adjacency matrices are VxV, the product matrix will also be VxV
		double C[][]=new double[V][V];    
	  
	        int i, j, k; 
	  
	        // Multiply the two matrices 
	        for (i = 0; i < V; i++) { 
	            for (j = 0; j < V; j++) { 
	                for (k = 0; k < V; k++) 
	                    C[i][j] += A[i][k] * B[k][j]; 
	            } 
	        } 

		return C;
	} 
	
	static int[][] hMatrixMultiplication(int[][] A, int[][] B)
	{
		// Matrix to store the result 
		// since adjacency matrices are VxV, the product matrix will also be VxV
		int C[][]=new int[V][V];    
	  
	        int i, j; 
	  
	        // Multiply the two matrices 
	        for (i = 0; i < V; i++) 
	        { 
	            for (j = 0; j < V; j++) 
	            { 
	                    C[i][j] = A[i][j] * B[i][j]; 
	            } 
	        } 

		return C;
	} 

	static int[][] matrixTranspose(int[][] A) 
	{ 
		// Matrix to store the result
		// since adjacency matrices are VxV, the transpose matrix will also be VxV
		int c[][]=new int[V][V]; 

	        int i, j; 
	        for (i = 0; i < V; i++) 
	            for (j = 0; j < V; j++) 
	                c[i][j] = A[j][i];

		return c; 
	}   
	
	static double[][] LaplacianMatrix(int[][] Dw,int[][] W)
	{
		double[][] Dexponent=new double[V][V];
		double[][] Lm=new double[V][V];
		int[][] I=new int[V][V];
		
		for(int i=0;i<V;i++)
			I[i][i]=1;
		for(int i=0;i<V;i++)
		{
			Dexponent[i][i]=1/(Math.sqrt(Dw[i][i]));	
		}
		
		Lm=matrixDoubleIntSubtraction(I,matrixDoubleDoubleMultiplication(matrixDoubleMultiplication(Dexponent,W),Dexponent));
		
		return Lm;
	}
}
