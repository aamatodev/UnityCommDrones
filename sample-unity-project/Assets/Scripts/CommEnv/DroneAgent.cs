using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Utils;

namespace CommEnv
{
    public class DroneAgent : Agent
    {
        [SerializeField] private GameObject baseStation1;
        [SerializeField] private GameObject baseStation2;

        [SerializeField] private Vector3 startingPos;

        private int _numOfConnections = 0;
        const double TOLERANCE = 0.1;
        
        protected override void Awake()
        {
            base.Awake();
            GameManager.Instance.RegisterAgent(this.transform.gameObject);
            
        }

        private void OnCollisionEnter(Collision other)
        {
            GameManager.Instance.RegisterConnection(this.transform.gameObject, other.gameObject);
        }

        private void OnCollisionExit(Collision other)
        {
            GameManager.Instance.RemoveConnection(this.transform.gameObject, other.gameObject);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            base.Heuristic(in actionsOut);
            
            var discreteActionsOut = actionsOut.DiscreteActions;
            //  discreteActionsOut[0] = k_NoAction;
        
            if (Input.GetKey(KeyCode.D))
            {
                discreteActionsOut[0] = (int)Key.k_right;
            }
            if (Input.GetKey(KeyCode.W))
            {
                discreteActionsOut[0] = (int)Key.k_up;
            }
            if (Input.GetKey(KeyCode.A))
            {
                discreteActionsOut[0] = (int)Key.k_left;
            }
            if (Input.GetKey(KeyCode.S))
            {
                discreteActionsOut[0] = (int)Key.k_down;
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);
            foreach (var a in GameManager.Instance.GetRegisteredAgents())
            {
                sensor.AddObservation(a.transform.position);
            }

        }

        public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            base.WriteDiscreteActionMask(actionMask);

           
            
            if (Math.Abs(this.transform.position.x - 9) < TOLERANCE)
            {
                actionMask.SetActionEnabled(0, (int)Key.k_up, false);
            }            
            
            if (Math.Abs(this.transform.position.x - (-9)) < TOLERANCE)
            {
                actionMask.SetActionEnabled(0, (int)Key.k_down, false);
            }            
            
            if (Math.Abs(this.transform.position.z - (-9)) < TOLERANCE)
            {
                actionMask.SetActionEnabled(0, (int)Key.k_left, false);
            }            
            
            if (Math.Abs(this.transform.position.z - 9) < TOLERANCE)
            {
                actionMask.SetActionEnabled(0, (int)Key.k_right, false);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var action = actions.DiscreteActions[0];

            var isActionValid = ValidateAction(action);
            
            if(isActionValid)
                PerformAction(action);
            
            ComputeReward();
            
            if (AreBaseStationsConnected())
            {
                GameManager.Instance.Reset();
            }            
            
        }

        private bool ValidateAction(int action)
        {
            var expectedPosition = this.transform.position;
            switch (action)
            {
                case (int)Key.k_noAction:
                    AddReward(-0.1f);
                    break;
                case (int)Key.k_up:
                    expectedPosition += new Vector3(1, 0, 0);
                    break;
                case (int)Key.k_down:
                    expectedPosition += new Vector3(-1, 0, 0);
                    break;
                case (int)Key.k_left:
                    expectedPosition += new Vector3(0, 0, -1);
                    break;
                case (int)Key.k_right:
                    expectedPosition += new Vector3(0, 0, 1);
                    break;
            }
            
            if(expectedPosition.x < -9 || expectedPosition.x > 9 || expectedPosition.z < -9 || expectedPosition.z > 9)
                return false;
            
            return true;
        }


        // perform action
        // 0 - dont move
        // 1 - move up
        // 2 - move down
        // 3 - move left
        // 4 - move right
        private void PerformAction(int action)
        {
            switch (action)
            {
                case (int)Key.k_noAction:
                    AddReward(-0.1f);
                    break;
                case (int)Key.k_up:
                    this.transform.position += new Vector3(1, 0, 0);
                    break;
                case (int)Key.k_down:
                    this.transform.position += new Vector3(-1, 0, 0);
                    break;
                case (int)Key.k_left:
                    this.transform.position += new Vector3(0, 0, -1);
                    break;
                case (int)Key.k_right:
                    this.transform.position += new Vector3(0, 0, 1);
                    break;
            }
        }

        private int oldReward = 0;
        // function to compute reward
        private void ComputeReward()
        {   
            _numOfConnections = GameManager.Instance.GetAgentDegree(this.transform.gameObject);

            var totalReward = _numOfConnections * 5 + (AreBaseStationsConnected() ? 100 : 0);
            if(transform.name == "Drone1" && oldReward != totalReward)
                Debug.Log("Total reward: " + totalReward);
            SetReward(totalReward);
        }

        private bool AreBaseStationsConnected()
        {
            return GameManager.Instance.AreBaseStationConnected();
        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            
            this.transform.position = startingPos;
        }
    }
}
