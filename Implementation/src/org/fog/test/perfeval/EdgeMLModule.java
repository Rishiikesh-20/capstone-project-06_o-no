package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.application.AppModule;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import org.fog.utils.GeoLocation;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.io.File;
import java.util.logging.Logger;


public class EdgeMLModule extends AppModule {
    private static final Logger LOGGER = Logger.getLogger(EdgeMLModule.class.getName());
    private String modelPath = IntelliPdM.projectDirPath + "/python_ml/ann_model.keras";
    private int hostDeviceId;

    public EdgeMLModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, CustomTupleScheduler scheduler, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", scheduler, new HashMap<>());
        this.hostDeviceId = hostDeviceId;
        ((CustomTupleScheduler) getCloudletScheduler()).setModule(this);
    }


    protected void processTupleArrival(Tuple tuple) {
        double startTime = CloudSim.clock();
        LOGGER.info("EdgeMLModule received tuple at time " + String.format("%.2f", CloudSim.clock()) + ": " + tuple.getTupleType());
        
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            int machineId = ((Number) data.getOrDefault("machine_id", 0)).intValue();
            
            if ("UPDATE_MODEL".equals(tuple.getTupleType())) {
                handleModelUpdate(data);
                return;
            }

            LOGGER.info("EdgeML processing data for Machine-" + machineId + " at time " + String.format("%.2f", CloudSim.clock()));
            handlePrediction(data, startTime);
        }
    }
    
    private void handleModelUpdate(Map<String, Object> data) {
        LOGGER.info(" EdgeML receiving model update at time " + String.format("%.2f", CloudSim.clock()));
        String base64Model = (String) data.get("model_base64");
        byte[] modelBytes = java.util.Base64.getDecoder().decode(base64Model);
        try (FileOutputStream fos = new FileOutputStream(modelPath)) {
            fos.write(modelBytes);
            MetricsCollector.recordModelUpdate();
            LOGGER.info("Edge model updated successfully at time " + String.format("%.2f", CloudSim.clock()));
        } catch (Exception e) {
            LOGGER.severe("Failed to update edge model: " + e.getMessage());
        }
    }
    
    private void handlePrediction(Map<String, Object> data, double startTime) {
        int machineId = ((Number) data.getOrDefault("machine_id", 0)).intValue();
        int trueFault = ((Number) data.getOrDefault("true_fault", 0)).intValue();
        double temp = ((Number) data.getOrDefault("temp", 50.0)).doubleValue();
        double voltage = ((Number) data.getOrDefault("voltage", 220.0)).doubleValue();
        
        LOGGER.info("EdgeML analyzing Machine-" + machineId + 
                   " (temp=" + String.format("%.1f", temp) + 
                   "C, voltage=" + String.format("%.1f", voltage) + 
                   "V, true_fault=" + trueFault + ")");

        String tempFile = IntelliPdM.projectDirPath + "/python_ml/temp_input_edge_" + UUID.randomUUID() + ".json";
        JSONObject json = new JSONObject();
        for (Map.Entry<String, Object> entry : data.entrySet()) {
            json.put(entry.getKey(), entry.getValue());
        }
        
        try (FileWriter fw = new FileWriter(tempFile)) {
            fw.write(json.toJSONString());
        } catch (Exception e) {
            LOGGER.severe("Failed to write temp input: " + e.getMessage());
            return;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(IntelliPdM.pythonExec, "python_ml/predict_ann.py", tempFile);
            pb.directory(new File(IntelliPdM.projectDirPath));
            Process p = pb.start();
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String output = br.readLine();
            
            if (output != null) {
                JSONParser parser = new JSONParser();
                JSONObject result = (JSONObject) parser.parse(output);
                double prob = ((Number) result.get("prob")).doubleValue();
                int fault = ((Number) result.get("fault")).intValue();
                double latency = ((Number) result.getOrDefault("latency_ms", 0.0)).doubleValue();
                String method = (String) result.getOrDefault("method", "ann");

                double totalLatency = (CloudSim.clock() - startTime) * 1000; 
                LOGGER.info("EdgeML Prediction for Machine-" + machineId + 
                           ": FAULT=" + (fault == 1 ? "YES" : "NO") + 
                           " (probability=" + String.format("%.3f", prob) + 
                           ", method=" + method + 
                           ", latency=" + String.format("%.2f", latency) + "ms)" +
                           " | Expected: " + (trueFault == 1 ? "FAULT" : "NORMAL"));

                MetricsCollector.recordEdgePrediction(fault, trueFault, latency, method, machineId);
                MetricsCollector.updateNetworkUsage(1000); 

                if (fault == 1) {
                    triggerActuator(machineId, prob, method);
                }
            }
        } catch (Exception e) {
            LOGGER.severe("ANN prediction failed: " + e.getMessage());
            handleFallbackPrediction(data, machineId, trueFault, startTime);
        } finally {
            new File(tempFile).delete(); 
        }
    }
    
    private void handleFallbackPrediction(Map<String, Object> data, int machineId, int trueFault, double startTime) {
        double temp = ((Number) data.get("temp")).doubleValue();
        double voltage = ((Number) data.get("voltage")).doubleValue();
        
        int fallbackFault = (temp > 65 || voltage > 250 || voltage < 200) ? 1 : 0;
        double prob = fallbackFault == 1 ? 0.8 : 0.2;
        double latency = (CloudSim.clock() - startTime) * 1000;
        
        LOGGER.warning("EdgeML Fallback Prediction for Machine-" + machineId + 
                      ": FAULT=" + (fallbackFault == 1 ? "YES" : "NO") + 
                      " (threshold-based, latency=" + String.format("%.2f", latency) + "ms)");
        
        MetricsCollector.recordEdgePrediction(fallbackFault, trueFault, latency, "fallback_threshold", machineId);
        
        if (fallbackFault == 1) {
            triggerActuator(machineId, prob, "fallback");
        }
    }
    
    private void triggerActuator(int machineId, double probability, String method) {
        String actuatorName = "actuator-" + machineId;
        try {
            Actuator actuator = IntelliPdM.actuators.stream()
                .filter(a -> a.getName().equals(actuatorName))
                .findFirst().get();
            
            DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0, 
                new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            stopTuple.setUserId(getUserId());
            stopTuple.setTupleType("STOP_ACTUATOR");
            stopTuple.setActuatorId(actuator.getId());
            sendTuple(stopTuple, "STOP_ACTUATOR");
            
            LOGGER.info(" EdgeML triggered STOP for Machine-" + machineId + 
                       " (probability=" + String.format("%.3f", probability) + 
                       ", method=" + method + ") at time " + String.format("%.2f", CloudSim.clock()));
                       
        } catch (Exception e) {
            LOGGER.severe("Failed to trigger actuator for Machine-" + machineId + ": " + e.getMessage());
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        tuple.setDestModuleName(destModule);
        CloudSim.send(hostDeviceId, hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
        LOGGER.info("EdgeMLModule sent tuple to " + destModule + " at time " + String.format("%.2f", CloudSim.clock()));
    }
}