using System.Collections.Generic;

namespace CommEnv.Utils
{
    class Graph
    {
        private Dictionary<int, List<int>> adjacencyList;

        public Graph()
        {
            adjacencyList = new Dictionary<int, List<int>>();
        }
        
        public List<int> GetNodes()
        {
            return new List<int>(adjacencyList.Keys);
        }

        // Add an edge to the graph
        public void AddEdge(int source, int destination)
        {
            if (!adjacencyList.ContainsKey(source))
            {
                adjacencyList[source] = new List<int>();
            }
            adjacencyList[source].Add(destination);

            if (!adjacencyList.ContainsKey(destination))
            {
                adjacencyList[destination] = new List<int>();
            }
            adjacencyList[destination].Add(source);
        }
        
        // Remove an edge from the graph
        public void RemoveEdge(int source, int destination)
        {
            if (adjacencyList.ContainsKey(source))
            {
                adjacencyList[source].Remove(destination);
            }
            if (adjacencyList.ContainsKey(destination))
            {
                adjacencyList[destination].Remove(source);
            }
        }
        
        
        // Check if two nodes are connected
        public bool AreNodesConnected(int node1, int node2)
        {
            HashSet<int> visited = new HashSet<int>();
            return DFS(node1, node2, visited);
        }

        // Get the degree (number of connections) of a node
        public int GetNodeDegree(int node)
        {
            if (adjacencyList.ContainsKey(node))
            {
                return adjacencyList[node].Count;
            }
            return 0;
        }

        private bool DFS(int current, int target, HashSet<int> visited)
        {
            if (current == target)
            {
                return true;
            }

            visited.Add(current);

            if (adjacencyList.ContainsKey(current))
            {
                foreach (var neighbor in adjacencyList[current])
                {
                    if (!visited.Contains(neighbor))
                    {
                        if (DFS(neighbor, target, visited))
                        {
                            return true;
                        }
                    }
                }
            }

            return false;
        }
    }
}