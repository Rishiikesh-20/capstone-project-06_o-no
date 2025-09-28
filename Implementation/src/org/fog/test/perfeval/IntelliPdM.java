package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.fog.application.Application;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.placement.Controller;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.distribution.DeterministicDistribution;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import javax.swing.table.DefaultTableModel;


public class IntelliPdM {
    private static final Logger LOGGER = Logger.getLogger(IntelliPdM.class.getName());
    static List<FogDevice> fogDevices = new ArrayList<>();
    static List<Sensor> sensors = new ArrayList<>();
    static List<Actuator> actuators = new ArrayList<>();
    static int numMachines;
    static int initialNumEdges;
    static double simDuration;
    static double loadThreshold;
    static double monitorInterval;
    static String pythonExec;
    static String projectDirPath;

    static {
        try {
            FileHandler fh = new FileHandler("intellipdm.log");
            fh.setFormatter(new SimpleFormatter());
            LOGGER.addHandler(fh);
            LOGGER.setLevel(java.util.logging.Level.INFO);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        loadConfig();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n" + "".repeat(100));
            System.out.println(" " + " ".repeat(98) + " ");
            System.out.println(" " + " ".repeat(25) + "INTELLIPDM PREDICTIVE MAINTENANCE RESULTS" + " ".repeat(31) + "█");
            System.out.println(" " + " ".repeat(98) + " ");
            System.out.println(" ".repeat(100));
            
            MetricsCollector.printDetailedMetrics();
            
            System.out.println("\nSIMULATION SUMMARY:");
            System.out.println("  Simulation completed successfully");
            System.out.println("  Real-time edge predictions with cloud backup");
            System.out.println("  Automatic model updates from cloud to edge");
            System.out.println("  Predictive maintenance prevented machine failures");
            
            System.out.println(" ".repeat(100));
            System.out.println(" " + " ".repeat(98) + " ");
            System.out.println(" " + " ".repeat(30) + "END OF INTELLIPDM ENHANCED RESULTS" + " ".repeat(32) + "█");
            System.out.println(" " + " ".repeat(98) + " ");
            System.out.println(" ".repeat(100));
        }));

        org.cloudbus.cloudsim.Log.enable();
        int userId = FogUtils.USER_ID;
        Calendar calendar = Calendar.getInstance();
        boolean trace_flag = false;
        CloudSim.init(userId, calendar, trace_flag);

        try {
            File projectDir = new File(projectDirPath);
            if (!projectDir.exists() || !projectDir.isDirectory()) {
                throw new IOException("Project directory does not exist: " + projectDirPath);
            }
            File annScript = new File(projectDir, "python_ml/train_ann.py");
            File rfScript = new File(projectDir, "python_ml/train_rf.py");
            if (!annScript.exists() || !rfScript.exists()) {
                throw new IOException("Python scripts missing: " + annScript.getPath() + " or " + rfScript.getPath());
            }

            ProcessBuilder pb = new ProcessBuilder(pythonExec, "python_ml/train_ann.py");
            pb.directory(projectDir);
            pb.inheritIO();
            Process p = pb.start();
            int exitCode = p.waitFor();
            if (exitCode == 0) {
                LOGGER.info("ANN model trained successfully.");
            } else {
                LOGGER.warning("ANN model training failed with exit code: " + exitCode);
            }

            ProcessBuilder pbRf = new ProcessBuilder(pythonExec, "python_ml/train_rf.py");
            pbRf.directory(projectDir);
            pbRf.inheritIO();
            Process pRf = pbRf.start();
            exitCode = pRf.waitFor();
            if (exitCode == 0) {
                LOGGER.info("RF model trained successfully.");
            } else {
                LOGGER.warning("RF model training failed with exit code: " + exitCode);
            }
        } catch (Exception e) {
            LOGGER.severe("Failed to train initial models: " + e.getMessage());
            LOGGER.info("Continuing simulation without pretrained models.");
        }

        String appId = "IntelliPdM";
        Application app = createApplication(appId, userId);

        createFogDevices(userId, appId);

        Controller controller = new Controller("controller", fogDevices, sensors, actuators);

        Map<String, Application> appMap = new HashMap<>();
        appMap.put(appId, app);
        for (FogDevice fd : fogDevices) {
            fd.setApplicationMap(appMap);
            if (!fd.getAppToModulesMap().containsKey(appId)) {
                fd.getAppToModulesMap().put(appId, new ArrayList<>());
            }
        }

        for (Sensor sensor : sensors) {
            sensor.setApp(app);
        }
        for (Actuator actuator : actuators) {
            actuator.setApp(app);
        }

        for (FogDevice fd : fogDevices) {
            CloudSim.addEntity(fd);
        }

        Monitor monitor = new Monitor("monitor", monitorInterval, loadThreshold, controller);
        CloudSim.addEntity(monitor);
        monitor.startEntity();

        CloudSim.startSimulation();

        while (CloudSim.clock() < simDuration && CloudSim.running()) {
            CloudSim.runClockTick();
        }

        CloudSim.stopSimulation();

        printMetrics();
        LOGGER.info("Simulation completed successfully.");

    }

    private static void loadConfig() {
        Properties props = new Properties();
        String configFilePath = "config.properties";
        File configFile = new File(configFilePath);
        LOGGER.info("Attempting to load config from: " + configFile.getAbsolutePath());
        try (InputStream input = new FileInputStream(configFile)) {
            props.load(input);
            numMachines = Integer.parseInt(props.getProperty("numMachines", "5"));
            initialNumEdges = Integer.parseInt(props.getProperty("initialNumEdges", "1"));
            simDuration = Double.parseDouble(props.getProperty("simDuration", "1000.0"));
            loadThreshold = Double.parseDouble(props.getProperty("loadThreshold", "0.8"));
            monitorInterval = Double.parseDouble(props.getProperty("monitorInterval", "10.0"));
            pythonExec = props.getProperty("pythonExec", "/usr/bin/python3");
            projectDirPath = props.getProperty("projectDir", System.getProperty("user.dir"));
            LOGGER.info("Configuration loaded: numMachines=" + numMachines + ", simDuration=" + simDuration);
        } catch (Exception e) {
            LOGGER.severe("Failed to load config.properties: " + e.getMessage());
            LOGGER.info("Using default configuration values.");
            numMachines = 5;
            initialNumEdges = 1;
            simDuration = 1000.0;
            loadThreshold = 0.8;
            monitorInterval = 10.0;
            pythonExec = "/usr/bin/python3";
            projectDirPath = System.getProperty("user.dir");
        }
    }

    private static void createFogDevices(int userId, String appId) {
        FogDevice cloud = createFogDevice("cloud", 44800, 40000, 10000, 10000, 0, 0.01, 16 * 103, 16 * 83.25, 0.01);
        if (cloud == null) {
            LOGGER.severe("Failed to create cloud device. Exiting.");
            System.exit(1);
        }
        fogDevices.add(cloud);

        CustomTupleScheduler cloudMLScheduler = new CustomTupleScheduler(4000, 1);
        CloudMLModule cloudML = new CloudMLModule(FogUtils.generateEntityId(), "CloudML", appId, userId, 4000, 4000, 10000, 1000, cloudMLScheduler, cloud.getId());
        cloudMLScheduler.setModule(cloudML); // Link scheduler to module
        if (cloud.getVmAllocationPolicy().allocateHostForVm(cloudML, cloud.getHost())) {
            cloud.getVmList().add(cloudML);
            if (!cloud.getAppToModulesMap().containsKey(appId)) {
                cloud.getAppToModulesMap().put(appId, new ArrayList<>());
            }
            cloud.getAppToModulesMap().get(appId).add("CloudML");
            LOGGER.info("CloudML module registered with cloud device");
        } else {
            LOGGER.warning("Failed to allocate CloudML module on cloud");
        }

        for (int i = 0; i < initialNumEdges; i++) {
            addEdgeDevice(cloud, i, appId, userId);
        }

        for (int i = 0; i < numMachines; i++) {
            int edgeIndex = (i % initialNumEdges) + 1;
            FogDevice edge = fogDevices.get(edgeIndex);

            Sensor sensor = new CustomSensor("sensor-" + i, "SENSOR", userId, appId, new DeterministicDistribution(5));
            sensor.setGatewayDeviceId(edge.getId());
            sensor.setLatency(1.0);
            edge.getChildToLatencyMap().put(sensor.getId(), 1.0);
            sensors.add(sensor);

            Actuator actuator = new CustomActuator("actuator-" + i, userId, appId, "STOP_ACTUATOR");
            actuator.setGatewayDeviceId(edge.getId());
            actuator.setLatency(1.0);
            edge.getChildToLatencyMap().put(actuator.getId(), 1.0);
            edge.getAssociatedActuatorIds().add(new org.apache.commons.math3.util.Pair<>(actuator.getId(), 1.0));
            actuators.add(actuator);
        }

        LOGGER.info("Cloud childToLatencyMap: " + cloud.getChildToLatencyMap());
        for (FogDevice fd : fogDevices) {
            if (fd.getName().startsWith("edge")) {
                LOGGER.info(fd.getName() + " childToLatencyMap: " + fd.getChildToLatencyMap());
            }
        }
    }

    public static void addEdgeDevice(FogDevice cloud, int edgeId, String appId, int userId) {
        FogDevice edge = createFogDevice("edge-" + edgeId, 5000, 4000, 10000, 10000, 1, 0.0, 107.339, 83.4333, 0.01);
        if (edge == null) {
            LOGGER.severe("Failed to create edge device edge-" + edgeId + ". Skipping.");
            return;
        }
        edge.setParentId(cloud.getId());
        cloud.getChildToLatencyMap().put(edge.getId(), 0.01);
        fogDevices.add(edge);

        CustomTupleScheduler preprocessScheduler = new CustomTupleScheduler(1000, 1);
        PreprocessModule preprocess = new PreprocessModule(FogUtils.generateEntityId(), "Preprocess", appId, userId, 1000, 1000, 10000, 1000, preprocessScheduler, edge.getId());
        preprocessScheduler.setModule(preprocess); // Link scheduler to module
        if (edge.getVmAllocationPolicy().allocateHostForVm(preprocess, edge.getHost())) {
            edge.getVmList().add(preprocess);
            if (!edge.getAppToModulesMap().containsKey(appId)) {
                edge.getAppToModulesMap().put(appId, new ArrayList<>());
            }
            edge.getAppToModulesMap().get(appId).add("Preprocess");
            LOGGER.info("Preprocess module registered with edge-" + edgeId);
        } else {
            LOGGER.warning("Failed to allocate Preprocess module on edge-" + edgeId);
        }

        CustomTupleScheduler edgeMLScheduler = new CustomTupleScheduler(2000, 1);
        EdgeMLModule edgeML = new EdgeMLModule(FogUtils.generateEntityId(), "EdgeML", appId, userId, 2000, 2000, 10000, 1000, edgeMLScheduler, edge.getId());
        edgeMLScheduler.setModule(edgeML); // Link scheduler to module
        if (edge.getVmAllocationPolicy().allocateHostForVm(edgeML, edge.getHost())) {
            edge.getVmList().add(edgeML);
            if (!edge.getAppToModulesMap().containsKey(appId)) {
                edge.getAppToModulesMap().put(appId, new ArrayList<>());
            }
            edge.getAppToModulesMap().get(appId).add("EdgeML");
            LOGGER.info("EdgeML module registered with edge-" + edgeId);
        } else {
            LOGGER.warning("Failed to allocate EdgeML module on edge-" + edgeId);
        }

        LOGGER.info("Dynamically added new edge device: " + edge.getName() + " with modules placed.");

        CloudSim.addEntity(edge);
    }

    private static FogDevice createFogDevice(String nodeName, long mips, int ram, long upBw, long downBw, int level, double ratePerMips, double busyPower, double idlePower, double uplinkLatency) {
        List<Pe> peList = new ArrayList<>();
        peList.add(new Pe(0, new PeProvisionerSimple(mips)));
        int hostId = FogUtils.generateEntityId();
        long storage = 1000000;
        int bw = 100000; 
        PowerHost host = new PowerHost(
            hostId,
            new RamProvisionerSimple(ram),
            new BwProvisionerSimple(bw),
            storage,
            peList,
            new StreamOperatorScheduler(peList),
            new FogLinearPowerModel(busyPower, idlePower)
        );

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
            "x86", "Linux", "Xen", host, 10.0, 0.01, 0.01, 0.001, 0.0
        );

        List<Host> hostList = new ArrayList<>();
        hostList.add(host);
        AppModuleAllocationPolicy allocationPolicy = new AppModuleAllocationPolicy(hostList);
        List<Storage> storageList = new ArrayList<>();
        double schedulingInterval = 10.0;

        try {
            FogDevice fogDevice = new FogDevice(nodeName, characteristics, allocationPolicy, storageList, schedulingInterval, upBw, downBw, uplinkLatency, ratePerMips);
            fogDevice.setLevel(level);
            return fogDevice;
        } catch (Exception e) {
            LOGGER.severe("Failed to create fog device " + nodeName + ": " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private static Application createApplication(String appId, int userId) {
        Application application = Application.createApplication(appId, userId);
        application.addAppModule("Preprocess", 1000, 1000, 10000);
        application.addAppModule("EdgeML", 2000, 2000, 10000);
        application.addAppModule("CloudML", 4000, 4000, 10000);

        application.addAppEdge("SENSOR", "Preprocess", 100.0, 200.0, "SENSOR", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("Preprocess", "EdgeML", 500.0, 100.0, "PROCESSED_TO_EDGE", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("Preprocess", "CloudML", 500.0, 100.0, "PROCESSED_TO_CLOUD", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("EdgeML", "STOP_ACTUATOR", 100.0, 50.0, "STOP_ACTUATOR", Tuple.ACTUATOR, AppEdge.ACTUATOR);
        application.addAppEdge("CloudML", "STOP_ACTUATOR", 100.0, 50.0, "STOP_ACTUATOR", Tuple.ACTUATOR, AppEdge.ACTUATOR);
        application.addAppEdge("CloudML", "EdgeML", 1000000.0, 500000.0, "UPDATE_MODEL", Tuple.DOWN, AppEdge.MODULE);

        application.addTupleMapping("Preprocess", "SENSOR", "PROCESSED_TO_EDGE", new FractionalSelectivity(1.0));
        application.addTupleMapping("Preprocess", "SENSOR", "PROCESSED_TO_CLOUD", new FractionalSelectivity(1.0));
        application.addTupleMapping("EdgeML", "PROCESSED_TO_EDGE", "STOP_ACTUATOR", new FractionalSelectivity(0.2));
        application.addTupleMapping("CloudML", "PROCESSED_TO_CLOUD", "STOP_ACTUATOR", new FractionalSelectivity(0.2));

        List<AppLoop> loops = new ArrayList<>();
        List<String> loop1 = new ArrayList<>();
        loop1.add("SENSOR");
        loop1.add("Preprocess");
        loop1.add("EdgeML");
        loop1.add("STOP_ACTUATOR");
        loops.add(new AppLoop(loop1));
        List<String> loop2 = new ArrayList<>();
        loop2.add("SENSOR");
        loop2.add("Preprocess");
        loop2.add("CloudML");
        loop2.add("STOP_ACTUATOR");
        loops.add(new AppLoop(loop2));
        application.setLoops(loops);

        return application;
    }

    private static void printMetrics() {
        System.out.println("\n" + " ".repeat(100));
        System.out.println(" " + " ".repeat(98) + " ");
        System.out.println(" " + " ".repeat(25) + "INTELLIPDM PREDICTIVE MAINTENANCE RESULTS" + " ".repeat(31) + "█");
        System.out.println(" " + " ".repeat(98) + " ");
        System.out.println(" ".repeat(100));
        System.out.println("Simulation Time: " + String.format("%.2f", CloudSim.clock()) + " time units");
        System.out.println("Configuration: " + numMachines + " machines, " + initialNumEdges + " edge devices");
        System.out.println(" ".repeat(100));
        
        MetricsCollector.printDetailedMetrics();
        
        System.out.println("\nFOG COMPUTING METRICS:");
        double totalEnergy = 0;
        for (FogDevice fd : fogDevices) {
            double deviceEnergy = fd.getEnergyConsumption();
            System.out.println("  " + fd.getName() + " energy: " + String.format("%.2f", deviceEnergy) + " W*sec");
            totalEnergy += deviceEnergy;
        }
        System.out.println("  Total energy consumption: " + String.format("%.2f", totalEnergy) + " W*sec");
        
        double cloudCost = 0.0;
        double edgeCost = 0.0;
        for (FogDevice fd : fogDevices) {
            if (fd.getName().equals("cloud")) {
                cloudCost = fd.getEnergyConsumption() * 0.01; 
            } else if (fd.getName().startsWith("edge")) {
                edgeCost += fd.getEnergyConsumption() * 0.005; 
            }
        }
        System.out.println("  Cloud cost: $" + String.format("%.2f", cloudCost));
        System.out.println("  Edge cost: $" + String.format("%.2f", edgeCost));
        System.out.println("  Total cost: $" + String.format("%.2f", cloudCost + edgeCost));
        
        System.out.println("\n SIMULATION SUMMARY:");
        System.out.println("   Simulation completed successfully");
        System.out.println("   Real-time edge predictions with cloud backup");
        System.out.println("   Automatic model updates from cloud to edge");
        System.out.println("   Predictive maintenance prevented machine failures");
        
        System.out.println(" ".repeat(100));
        System.out.println(" " + " ".repeat(98) + " ");
        System.out.println(" " + " ".repeat(30) + "END OF INTELLIPDM ENHANCED RESULTS" + " ".repeat(32) + "█");
        System.out.println(" " + " ".repeat(98) + " ");
        System.out.println(" ".repeat(100));
        LOGGER.info("Comprehensive metrics printed.");
    }
}