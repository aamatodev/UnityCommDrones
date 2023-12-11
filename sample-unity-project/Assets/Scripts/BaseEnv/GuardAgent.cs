using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

public class GuardAgent : Agent
{
    [SerializeField] private GameObject target;
    [SerializeField] private GameObject prisoner;

    
    const int k_NoAction = 0;  // do nothing!
    const int k_Up = 1;
    const int k_Down = 2;
    const int k_Left = 3;
    const int k_Right = 4;

    const float _planeLimit = 3.5f;
    
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
      //  discreteActionsOut[0] = k_NoAction;
        
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = k_Right;
        }
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = k_Up;
        }
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = k_Left;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = k_Down;
        }
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);
        sensor.AddObservation(target.transform.position);
        sensor.AddObservation(transform.position);
        sensor.AddObservation(prisoner.transform.position);
    }

     public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // Mask the necessary actions if selected by the user.
    
        // Prevents the agent from picking an action that would make it collide with a wall
        var localPosition = transform.localPosition;
        var positionX = (int)localPosition.x;
        var positionZ = (int)localPosition.z;
        
            
        if (positionX <= -3.5f)
        {
            actionMask.SetActionEnabled(0, k_Left, false);
        }

        if (positionX >= 3.5f)
        {
            actionMask.SetActionEnabled(0, k_Right, false);
        }

        if (positionZ <= -3.5f)
        {
            actionMask.SetActionEnabled(0, k_Down, false);
        }

        if (positionZ >= 3.5f)
        {
            actionMask.SetActionEnabled(0, k_Up, false);
        }
        
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
         
        AddReward(-0.01f);
        var action = actions.DiscreteActions[0];

        
        switch (action)
        {
            case k_NoAction:
                Debug.Log("received k_NoAction");
                break;
            case k_Right:
                transform.position += new Vector3(0.5f, 0, 0f);
                break;
            case k_Left:
                transform.position += new Vector3(-0.5f, 0, 0f);
                break;
            case k_Up:
                transform.position += new Vector3(0f, 0, 0.5f);
                break;
            case k_Down:
                transform.position += new Vector3(0f, 0, -0.5f);
                break;
            default:
                throw new ArgumentException("Invalid action value");
        }
        
        
        if(transform.localPosition.x < -_planeLimit || transform.localPosition.z < -_planeLimit || transform.localPosition.x > _planeLimit || transform.localPosition.z > _planeLimit )
            EndEpisode();
        
        // Rewards
        float distanceToTarget = Vector3.Distance(prisoner.transform.localPosition, target.transform.localPosition);
        float distanceToPrisoner = Vector3.Distance(this.transform.localPosition, prisoner.transform.localPosition);
        

        if (distanceToPrisoner < 1.6f)
        {
            AddReward(10f);
            EndEpisode();
        }        
        
        if (distanceToTarget < 1.6f)
        {
            AddReward(-10f);
            EndEpisode();
        }
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        transform.position = new Vector3(Random.Range(-_planeLimit, _planeLimit), 0.5f,
            Random.Range(-_planeLimit, _planeLimit));
    }
}