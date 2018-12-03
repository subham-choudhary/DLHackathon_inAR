/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller for selecting images and applying Vision + Core ML processing.
*/

import UIKit
import CoreML
import Vision
import ImageIO
import AVFoundation
import ARKit

class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var classificationLabel: UILabel!
    @IBOutlet weak var lbl1: UILabel!
    @IBOutlet weak var lbl2: UILabel!
    @IBOutlet weak var lbl3: UILabel!
    @IBOutlet weak var lbl4: UILabel!
    @IBOutlet weak var veu1: UIView!
    @IBOutlet weak var veu2: UIView!
    @IBOutlet weak var veu3: UIView!
    @IBOutlet weak var veu4: UIView!
    @IBOutlet weak var addLabelButton: UIButton!
    
    @IBOutlet weak var in1: UIImageView!
    @IBOutlet weak var in2: UIImageView!
    @IBOutlet weak var in3: UIImageView!
    @IBOutlet weak var in4: UIImageView!
    @IBOutlet weak var in5: UIImageView!
    @IBOutlet weak var in6: UIImageView!
    @IBOutlet weak var in7: UIImageView!
    @IBOutlet weak var sceneView: ARSCNView!
    
    let captureSession = AVCaptureSession()
    var previewLayer: CALayer!
    var captureDevice: AVCaptureDevice!
    var shouldPerformClassification = true
    let configuration = ARWorldTrackingConfiguration()
    
    let bubbleDepth : Float = 0.01
    let confidenceLevel: Float = 0.2
    var classificationsx = [VNClassificationObservation]()
    let myQueue = DispatchQueue(label: "myQueue", qos: .userInitiated)
    var center = CGPoint()
    
    var pointerNode = SCNNode()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        center = self.view.center
        setupTapGesture()
        setupAR()
        self.view.bringSubview(toFront: classificationLabel)
    }
    
    
    
    @IBAction func buttonTapped(_ sender: Any) {
        print("sdfdsfs")
    }
 
    
    // MARK: - Image Classification
    
    /// - Tag: PerformRequests
    func updateClassifications(for image: UIImage) {
        print("0")
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    /// - Tag: MLModelSetup
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: ImageClassifier2().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    
    /// Updates the UI with the results of the classification.
    /// - Tag: ProcessClassifications
    func processClassifications(for request: VNRequest, error: Error?) {
            guard let results = request.results else {
                return
            }
            let classifications = results as! [VNClassificationObservation]
            classificationsx = classifications
        
            DispatchQueue.main.async {
                if classifications.isEmpty {
                } else {
                    self.activateAndFeedLabels(classifications: classifications )
                    if classifications.first?.confidence ?? 0 > self.confidenceLevel {
                        self.toggleIndicator(green: true)
                    } else {
                        self.toggleIndicator(green: false)
                    }
            }
            self.shouldPerformClassification = true
        }
    }
}

extension ImageClassificationViewController: ARSCNViewDelegate {
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        
    }
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        
        if shouldPerformClassification {
            shouldPerformClassification = false
            let imageFromArkitScene:UIImage = sceneView.snapshot()
            updateClassifications(for: imageFromArkitScene)
        }
        resetPointerNode()
    }
    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        
    }
}
////////////////////////
//Custom Functions
////////////////////
extension ImageClassificationViewController {
    
    
    func setupTapGesture() {
        let tapGesture1 = UITapGestureRecognizer(target: self, action: #selector(tapped1))
        let tapGesture2 = UITapGestureRecognizer(target: self, action: #selector(tapped2))
        let tapGesture3 = UITapGestureRecognizer(target: self, action: #selector(tapped3))
        let tapGesture4 = UITapGestureRecognizer(target: self, action: #selector(tapped4))
        veu1.addGestureRecognizer(tapGesture4)
        veu2.addGestureRecognizer(tapGesture3)
        veu3.addGestureRecognizer(tapGesture2)
        veu4.addGestureRecognizer(tapGesture1)
    }
    
    @objc func tapped1() {
        renderItemPosition(index: 0)
    }
    @objc func tapped2() {
        renderItemPosition(index: 1)
    }
    @objc func tapped3() {
        renderItemPosition(index: 2)
    }
    @objc func tapped4() {
        renderItemPosition(index: 3)
    }
    
    func renderItemPosition(index: Int) {
        let str = classificationsx[index].identifier
        let node : SCNNode = createNewBubbleParentNode(str, img: sceneView.snapshot())
        let transform = pointerNode.worldTransform
        node.position = SCNVector3Make(transform.m41,transform.m42,transform.m43)
        sceneView.scene.rootNode.addChildNode(node)
    }
    
    func setupAR() {
        
        self.sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints]
        self.sceneView.autoenablesDefaultLighting = true
        if #available(iOS 11.3, *) {
            configuration.planeDetection = [.horizontal,.vertical]
        } else {
            // Fallback on earlier versions
        }
        self.sceneView.session.run(configuration)
        
        pointerNode = SCNNode(geometry: SCNSphere(radius: 0.005))
        pointerNode.geometry?.firstMaterial?.diffuse.contents = UIColor.white
        pointerNode.geometry?.firstMaterial?.specular.contents = UIColor.black
        sceneView.scene.rootNode.addChildNode(pointerNode)
    }
    
    func resetPointerNode() {
        DispatchQueue.main.async {
            if let hitTestResult = self.sceneView.hitTest(self.center, types: [.featurePoint, .existingPlane]).first {
                
                let transform = hitTestResult.worldTransform
                let thirdColumn = transform.columns.3
//                DispatchQueue.main.async {
                    self.pointerNode.position = SCNVector3Make(thirdColumn.x, thirdColumn.y, thirdColumn.z)
//                }
            }
        }
    }
    
//    func removePointerNode() {
////        myQueue.async {
//            self.pointerNode.enumerateChildNodes { (node, _) in
//                node.removeFromParentNode()
//            }
////        }
//    }
    
    func toggleIndicator(green: Bool) {
        
            self.in2.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            self.in6.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            self.in3.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            self.in5.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            self.in4.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            self.in1.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            self.in7.image = green ? #imageLiteral(resourceName: "green-button-icon-png-33") : #imageLiteral(resourceName: "red-button-icon-png-3")
            }

    func activateAndFeedLabels(classifications: [VNClassificationObservation]) {
        
        self.veu1.isHidden = false
        self.veu2.isHidden = false
        self.veu3.isHidden = false
        self.veu4.isHidden = false
        self.lbl1.text = String(format: "  (%.2f) %@", classifications[0].confidence,  classifications[0].identifier)
        self.lbl2.text = String(format: "  (%.2f) %@", classifications[1].confidence,  classifications[1].identifier)
        self.lbl3.text = String(format: "  (%.2f) %@", classifications[2].confidence,  classifications[2].identifier)
        self.lbl4.text = String(format: "  (%.2f) %@", classifications[3].confidence,  classifications[3].identifier)
    }
    
    func createNewBubbleParentNode(_ text : String, img: UIImage) -> SCNNode {

        let billboardConstraint = SCNBillboardConstraint()
        billboardConstraint.freeAxes = SCNBillboardAxis.Y
        
        // BUBBLE-TEXT
        let bubble = SCNText(string: text, extrusionDepth: CGFloat(bubbleDepth))
        var font = UIFont(name: "Futura", size: 0.30)
        font = font?.withTraits(traits: .traitBold)
        bubble.font = font
        bubble.alignmentMode = kCAAlignmentCenter
        bubble.firstMaterial?.diffuse.contents = UIColor.blue
        bubble.firstMaterial?.specular.contents = UIColor.white
        bubble.firstMaterial?.isDoubleSided = true
        // bubble.flatness // setting this too low can cause crashes.
        bubble.chamferRadius = CGFloat(bubbleDepth)
        
        // BUBBLE NODE
        let (minBound, maxBound) = bubble.boundingBox
        let bubbleNode = SCNNode(geometry: bubble)
        // Centre Node - to Centre-Bottom point
        bubbleNode.pivot = SCNMatrix4MakeTranslation( (maxBound.x - minBound.x)/2, minBound.y, bubbleDepth/2)
        // Reduce default text size
        bubbleNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        
        // CENTRE POINT NODE
        let sphere = SCNSphere(radius: 0.01)
        sphere.firstMaterial?.diffuse.contents = UIColor.black
        let sphereNode = SCNNode(geometry: sphere)
        //line
        let line = SCNCylinder(radius: 0.002, height: 0.4)
        line.firstMaterial?.diffuse.contents = UIColor.blue
        let lineNode = SCNNode(geometry: line)
        
        let image = img
        let imageNodeP = SCNNode(geometry: SCNPlane(width: 0.096, height: 0.167))
        imageNodeP.geometry?.firstMaterial?.diffuse.contents = UIColor.blue
        let imageNode = SCNNode(geometry: SCNPlane(width: 0.09, height: 0.16))
        imageNode.position.z = imageNode.position.z + 0.001
        imageNode.geometry?.firstMaterial?.diffuse.contents = image
        imageNodeP.addChildNode(imageNode)
        
        
        // BUBBLE PARENT NODE
        let bubbleNodeParent = SCNNode()
        bubbleNodeParent.addChildNode(bubbleNode)
        bubbleNodeParent.addChildNode(sphereNode)
        bubbleNodeParent.addChildNode(lineNode)
        bubbleNodeParent.constraints = [billboardConstraint]
        lineNode.position.y = bubbleNode.position.y + 0.2
        bubbleNode.position.y = bubbleNode.position.y + 0.4
        bubbleNodeParent.addChildNode(imageNodeP)
        imageNodeP.position.y = imageNode.position.y + 0.55
        
        return bubbleNodeParent
    }
}

extension UIFont {
    func withTraits(traits:UIFontDescriptorSymbolicTraits...) -> UIFont {
        let descriptor = self.fontDescriptor.withSymbolicTraits(UIFontDescriptorSymbolicTraits(traits))
        return UIFont(descriptor: descriptor!, size: 0)
    }
}
