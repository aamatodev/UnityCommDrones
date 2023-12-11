using System;
using System.Collections.Generic;
using CommEnv.Utils;
using UnityEngine;

namespace CommEnv
{
    public class GameManager : MonoBehaviour
    {
        // The two main base stations
        [SerializeField] private GameObject baseStation1;
        [SerializeField] private GameObject baseStation2;
        
        // Connection graph
        private Graph _graph = new Graph();
        private Dictionary<GameObject, int> _agents = new Dictionary<GameObject, int>();
        
        // Public property to access the singleton instance
        private static GameManager _instance;
        public static GameManager Instance
        {
            get
            {
                // If the instance does not exist, create it
                if (_instance == null)
                {
                    // Look for an existing instance in the scene
                    _instance = FindObjectOfType<GameManager>();

                    // If no instance exists, create a new GameObject with the script attached
                    if (_instance == null)
                    {
                        GameObject singletonObject = new GameObject(typeof(GameManager).Name);
                        _instance = singletonObject.AddComponent<GameManager>();
                    }
                }

                // Return the instance
                return _instance;
            }
        }
        private void Awake()
        {
            
            // Ensure that only one instance of the singleton exists
            if (_instance != null && _instance != this)
            {
                Destroy(gameObject);
            }
            else
            {
                _instance = this;
                DontDestroyOnLoad(gameObject);
            }
            
            //Register the two base station in the game
            RegisterAgent(baseStation1);
            RegisterAgent(baseStation2);
        }

        // Register an agent in the game
        public void RegisterAgent(GameObject g)
        {
            var index = _instance._agents.Count;
            _instance._agents.Add(g, index);
        }
        
        // Register a connection between two agents
        public void RegisterConnection(GameObject g1, GameObject g2)
        {
            var index1 = _instance._agents[g1];
            var index2 = _instance._agents[g2];
            
            if(!AreNodesConnected(index1, index2) || !AreNodesConnected(index2, index1) )   
                _instance._graph.AddEdge(index1, index2);
        }
        
        // Remove a connection between two agents
        public void RemoveConnection(GameObject g1, GameObject g2)
        {
            var index1 = _instance._agents[g1];
            var index2 = _instance._agents[g2];
            
            if(AreNodesConnected(index1, index2) || AreNodesConnected(index2, index1) )   
                _instance._graph.RemoveEdge(index1, index2);
        }
        
        // Get the degree of an agent
        public int GetAgentDegree(GameObject g)
        {
            var index = _instance._agents[g];
            return _instance._graph.GetNodeDegree(index);
        }

        // Check if two agents are connected
        public bool AreNodesConnected(int index1, int index2)
        {
            return _instance._graph.AreNodesConnected(index1, index2);
        }

        // Check if the two main base are connected
        public bool AreBaseStationConnected()
        {
            var base1Index = _instance._agents[baseStation1];
            var base2Index = _instance._agents[baseStation2];
            
            return _instance._graph.AreNodesConnected(base1Index, base2Index);
        }

        public void Reset()
        {
            foreach (var agent in _agents)
            {
                if (agent.Key.TryGetComponent(out DroneAgent a))
                {
                    a.EndEpisode();
                }
            }
        }

        public List<GameObject> GetRegisteredAgents()
        {
            List<GameObject> agents = new List<GameObject>();
            
            foreach (var agent in _agents)
            {
                agents.Add(agent.Key);
            }

            return agents;
        }
    }
}
