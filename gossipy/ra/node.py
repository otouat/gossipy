from __future__ import annotations
import random
import numpy as np
from PIL import Image
from numpy.random import randint, normal, rand
from numpy import ndarray
from typing import Optional, List
from torch import Tensor
from typing import Any, Optional, Union, Dict, Tuple, Iterable
from .data import DataDispatcher
from collections import OrderedDict
from multipledispatch import dispatch
from . import CACHE, LOG
from .core import AntiEntropyProtocol, CreateModelMode, MessageType, Message, P2PNetwork
from .utils import choice_not_n
from .model.handler import ModelHandler, PartitionedTMH, SamplingTMH, WeightedTMH
from .model.sampling import TorchModelSampling
import random as ran
import sys
import json
import math
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os, shutil

folder = 'images_original2'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

folder = 'images_created2'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache License, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["GossipNode",
           "PassiveGradientInversionAttacker",
           "PassThroughNode",
           "CacheNeighNode",
           "SamplingBasedNode",
           "PartitioningBasedNode",
           "PENSNode"]




class GossipNode():
    global call_number 
    call_number = 0


    global occurence_num_victim 
    occurence_num_victim = [0 for _ in range(4)]

    global the_victim
    nombre_s = [0, 1, 3]

# Choix aléatoire
    the_victim = random.choice(nombre_s)
# attacker = random.randint(0, nodes_num-1)

    global occurence_num_each  
    occurence_num_each = [0 for _ in range(4)]


    global pur_grad_victim 
    pur_grad_victim = [0 for _ in range(4)]
    
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync: bool=True):
        r"""Class that represents a generic node in a gossip network. 

        A node is identified by its index and it is initialized with a fixed delay :math:`\Delta` that
        represents the idle time. The node can be either synchronous or asynchronous. In the former case,
        the node will time out exactly :math:`\Delta` time steps into the round. Thus it is assumed that 
        :math:`0 < \Delta <` `round_len`. In the latter case, the node will time out to every 
        :math:`\Delta` time steps. In the synchronous case, :math:`\Delta \sim U(0, R)`, otherwise 
        :math:`\Delta \sim \mathcal{N}(R, R/10)` where :math:`R` is the round length.

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """
        self.idx: int = idx
        self.data:  Union[Tuple[Tensor, Optional[Tensor]], Tuple[ndarray, Optional[ndarray]]] = data
        self.round_len: int = round_len
        self.model_handler: ModelHandler = model_handler
        self.sync: bool = sync
        self.delta: int = randint(0, round_len) if sync else int(normal(round_len, round_len/10))
        self.p2p_net = p2p_net
        self.prob_n = [0,1,2,3]

      

        

    def init_model(self, local_train: bool=True, *args, **kwargs) -> None:
        """Initializes the local model.

        Parameters
        ----------
        local_train : bool, default=True
            Whether the local model should be trained for with the local data after the
            initialization.
        """

        self.model_handler.init()
        if local_train:
            self.model_handler._update(self.data[0])
         




    def get_peer(self) -> int:
        """Picks a random peer from the reachable nodes.

        Returns
        -------
        int
            The index of the randomly selected peer.
        """

        peers = self.p2p_net.get_peers(self.idx)
        if not peers:
            LOG.warning("Node %d has no peers.", self.idx)
            return None
        return random.choice(peers) if peers else choice_not_n(0, self.p2p_net.size(), self.idx)
        
    def timed_out(self, t: int) -> bool:
        """Checks whether the node has timed out.
        
        Parameters
        ----------
        t : int
            The current timestamp.
        
        Returns
        -------
        bool
            Whether the node has timed out.
        """

        return ((t % self.round_len) == self.delta) if self.sync else ((t % self.delta) == 0)

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Message:
        """Sends a message to the specified peer.

        The method actually prepares the message that will be sent to the peer.
        The sending is performed by the simluator and it may be delayed or it can fail.

        Parameters
        ----------
        t : int
            The current timestamp.
        peer : int
            The index of the peer node.
        protocol : AntiEntropyProtocol
            The protocol used to send the message.

        Returns
        -------
        Message
            The message to send.
        
        Raises
        ------
        ValueError
            If the protocol is not supported.
        
        See Also
        --------
        :class:`gossipy.simul.GossipSimulator`
        """
  
        
        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, peer, MessageType.PUSH, (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, peer, MessageType.PUSH_PULL, (key,))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    
    
    # @dispatch( int, Message)
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        """Receives a message from the peer.

        After the message is received, the local model is updated and merged according to the `mode`
        of the model handler. In case of a pull/push-pull message, the local model is sent back to the
        peer.

        Parameters
        ----------
        t : int
            The current timestamp.
        msg : Message
            The received message.
        
        Returns
        -------
        Message or `None`
            The message to be sent back to the peer. If `None`, there is no message to be sent back.
        """
   
        global images
        global call_number 

        call_number = call_number +1
        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value[0] if msg.value else None

        # neighbor = [0,1,2]
        # victim = 2
        # attacker = 3

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model = CACHE.pop(recv_model)

        

            self.model_handler(recv_model, self.data[0])

   

        global occurence_num_each
        global occurence_num_victim
        global the_victim
        global pur_grad_victim

        
        index_of_sender = self.prob_n.index(msg.sender)
        if self.idx == the_victim:
            occurence_num_victim[index_of_sender] = occurence_num_victim[index_of_sender] + 1
            pur_grad_victim.append(self.model_handler.model.state_dict())

        
        occurence_num_each[self.idx] = occurence_num_each[self.idx] + 1


        
        num_files = len([name for name in os.listdir('images_original2') if os.path.isfile(os.path.join('images_original2', name))])
        if num_files != len(self.p2p_net.get_peers(self.idx))+1:       
            axi = plt.subplot()
            axi.imshow(np.transpose(self.model_handler.data_used[0], [1, 2, 0]))
            # axi.set_title("original image" ) 
            plt.savefig(f'images_original2/{self.idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
          
            return Message(t, self.idx, msg.sender, MessageType.REPLY, (key,))
        return None
    



    def evaluate(self, ext_data: Optional[Any]=None) -> Dict[str, float]:
        """Evaluates the local model.

        Parameters
        ----------
        ext_data : Any, default=None
            The data to be used for evaluation. If `None`, the local test data will be used.
        
        Returns
        -------
        dict[str, float]
            The evaluation results. The keys are the names of the metrics and the values are
            the corresponding values.
        """

        if ext_data is None:
            return self.model_handler.evaluate(self.data[1])
        else:
            return self.model_handler.evaluate(ext_data)
    
    #CHECK: we need a more sensible check
    def has_test(self) -> bool:
        """Checks whether the node has a test set.

        Returns
        -------
        bool
            Whether the node has a test set.
        """

        if isinstance(self.data, tuple):
            return self.data[1] is not None
        else: return True
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} #{self.idx} (Δ={self.delta})"


    @classmethod
    def generate(cls,
                 data_dispatcher: DataDispatcher,
                 p2p_net: P2PNetwork,
                 model_proto: ModelHandler,
                 round_len: int,
                 sync: bool,
                 **kwargs) -> Dict[int, GossipNode]:
        """Generates a set of nodes.

        Parameters
        ----------
        data_dispatcher : DataDispatcher
            The data dispatcher used to distribute the data among the nodes.
        p2p_net : P2PNetwork
            The peer-to-peer network topology.
        model_proto : ModelHandler
            The model handler prototype.
        round_len : int
            The length of a round in time units.
        sync : bool
            Whether the nodes are synchronized with the round length or not.

        Returns
        -------
        Dict[int, GossipNode]
            The generated nodes.
        """
        
        nodes = {}
        for idx in range(p2p_net.size()):
            node = cls(idx=idx,
                       data=data_dispatcher[idx], 
                       round_len=round_len, 
                       model_handler=model_proto.copy(), 
                       p2p_net=p2p_net, 
                       sync=sync, 
                       **kwargs)
            nodes[idx] = node
        return nodes
    
    
    @classmethod
    def generate_with_adversaries(cls,
                                  cls2, # instantiation class of adversary (eg, PassiveGradientInversionAttacker class ( to be created in node.py))
                                  idx_adversaries : ndarray[int], # identifiers of adversaries
                 data_dispatcher: DataDispatcher,
                 p2p_net: P2PNetwork,
                 model_proto: ModelHandler,
                 round_len: int,
                 sync: bool,
                 **kwargs) -> Dict[int, GossipNode]:
        
        nodes = {}

        for idx in range(p2p_net.size()):
            if idx in idx_adversaries:
                node = cls2(idx=idx,
                       data=data_dispatcher[idx], 
                       round_len=round_len, 
                       model_handler=model_proto.copy(), 
                       p2p_net=p2p_net, 
                       sync=sync, 
                       **kwargs)
    
            else:
                node = cls(idx=idx,
                       data=data_dispatcher[idx], 
                       round_len=round_len, 
                       model_handler=model_proto.copy(), 
                       p2p_net=p2p_net, 
                       sync=sync, 
                       **kwargs)
            nodes[idx] = node
        return nodes
    


class PassiveGradientInversionAttacker(GossipNode):
    def __init__(self, idx, data, round_len, model_handler, p2p_net, sync, **kwargs):
        super().__init__(idx=idx, data=data, round_len=round_len, model_handler=model_handler, p2p_net=p2p_net, sync=sync, **kwargs)

        self.attack_neighbor = self.p2p_net.get_peers(idx)
        # self.victim = ran.choice(self.attack_neighbor) if self.attack_neighbor else None 
        self.victim = the_victim 
        
        self.neighbor = self.p2p_net.get_peers(self.victim)



        self.neighbor.append(self.victim)
        self.neighbor.sort()

        taille = len(self.neighbor) 
        valeur_defaut = None
        self.nodes_values = self.neighbor[:]
        self.nodes_array = [valeur_defaut for _ in range(taille)]
        self.occurence_num_attacker = [0 for _ in range(taille)]



       

    # def receive(self, t: int, msg: Message, my_neighbor: List[int], my_victim : int, the_attacker : int) -> Union[Message, None]:      
    def receive(self, t: int, msg: Message) -> Union[Message, None]:      
        global images
        global call_number 

        call_number = call_number +1

        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value[0] if msg.value else None
        self.token = 0


        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model = CACHE.pop(recv_model)

       
        
        self.model_handler(recv_model, self.data[0])
        

        
        num_files = len([name for name in os.listdir('images_original2') if os.path.isfile(os.path.join('images_original2', name))])
        if num_files != len(self.p2p_net.get_peers(self.idx))+1:       
            ax = plt.subplot()
            ax.imshow(np.transpose(self.model_handler.data_used[0], [1, 2, 0]))
            # ax.set_title("original image") 
            plt.savefig(f'images_original2/{self.idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

     
        # self.last_modified.append(msg.sender)

        global occurence_num_each
        global occurence_num_victim
        global pur_grad_victim
# i need to add this condition to make the attack better
        # and self.last_modified[len(self.last_modified)-2] == self.victim
        if len(self.nodes_values) != (len(self.neighbor)+1) :
            if msg.sender == self.victim and not None in self.nodes_array  :
                self.nodes_values.append(self.victim)
                self.nodes_array.append(recv_model.model.state_dict())
                

                

            else:
                index_l = self.nodes_values.index(msg.sender)
                self.nodes_array[index_l] = recv_model.model.state_dict()
                self.nodes_array[self.nodes_values.index(self.idx)] = self.model_handler.model.state_dict()
                if msg.sender == self.victim:
                    occurence_num_victim = [0 for _ in range(4)]
                    self.occurence_num_attacker = [0 for _ in range(4)]
                    occurence_num_each= [0 for _ in range(4)]
                  
            self.occurence_num_attacker[self.nodes_values.index(msg.sender)] += 1
            occurence_num_each[self.idx] = occurence_num_each[self.idx] + 1


        

        def sum_nested_structures_and_negate(structures):
            # Vérifie que 'structures' est une liste (ou un iterable) contenant au moins un élément
            if not structures or not all(isinstance(s, OrderedDict) for s in structures):
                raise ValueError("Le paramètre 'structures' doit être une liste d'OrderedDicts avec des valeurs tensorielles")            
            # Initialisation du résultat avec une copie profonde du premier élément pour garder les clés
            result = OrderedDict()
            for key in structures[0]:
                result[key] = 0

            # Itération sur chaque structure pour effectuer la somme
            for structure in structures[:-1]:
                for key in structure:
                    # Accumulation des sommes des tenseurs pour chaque clé
                    result[key] += structure[key]            
            # Négation des résultats accumulés
            for key in result:
                result[key] *= (1/ (len(self.nodes_array)-2) )        
            return result
        

        def invert_fully_g(gw, gb,  i=None ):
            # try:
            #     b = (1 / gb)
            # except RuntimeError:
            #     print("bias is null")
            #     return None
              

            
            b = ( (1. / (gb.numpy()[np.newaxis,:])))                                
            w = gw.numpy()
            # if not i is None:
            xa = b[:, 1] * w[1, :]
            # else:
            # xa = (np.matmul(b, w))
            print(b.shape, w.shape, xa.shape)
            # xa = xa.reshape(3,96,96)
            xa = xa.reshape(3,32,32)

            return normalize_img(xa)

       
        def normalize_img(x):
            x_min = x.min()
            x_max = x.max()
            range_ = x_max - x_min
            if range_ != 0:
                x_normalized = (x - x_min) / range_
            else:
                x_normalized = np.zeros(x.shape)

            return x_normalized

        def calculate_ssim(image1, image2):
            """Calculate the Structural Similarity Index (SSIM) between two images."""
            # Ensure images are the same size
            if image1.size != image2.size:
                raise ValueError("Images must have the same dimensions.")
            
            # Convert images to grayscale for SSIM calculation
            image1_gray = np.array(image1.convert('L'))
            image2_gray = np.array(image2.convert('L'))
            
            # Calculate SSIM
            ssim_index = ssim(image1_gray, image2_gray)
            return ssim_index


        def compare_images(single_image_path, directory_with_images):
            ssim_values = []
            image_files = []
            
            image1 = Image.open(single_image_path)
            
            for image_file in os.listdir(directory_with_images):
                if image_file.endswith('.png'):
                    image_path = os.path.join(directory_with_images, image_file)
                    image2 = Image.open(image_path)
                    ssim_value = calculate_ssim(image1, image2)
                    ssim_values.append(ssim_value)
                    image_files.append(image_file)
            
            # Normalize SSIM values for color mapping
        
            norm_ssim_values = (np.array(ssim_values) - min(ssim_values)) / (max(ssim_values) - min(ssim_values))
            colors = [plt.cm.RdYlGn(x) for x in norm_ssim_values]
            
            # Find the index of the best match
            max_ssim_index = np.argmax(ssim_values)
            best_match_image_path = os.path.join(directory_with_images, image_files[max_ssim_index])
            best_match_image = Image.open(best_match_image_path)

            # Plotting
            plt.figure(figsize=(14, 6))
            
            # SSIM values with colored bars
            plt.subplot(1, 3, 1)
            bars = plt.bar(image_files, ssim_values, color=colors)
            plt.xlabel('Image File')
            plt.ylabel('SSIM Value')
            plt.title('SSIM Comparison')
            plt.xticks(rotation=45)
            
            # Reference image
            plt.subplot(1, 3, 2)
            plt.imshow(image1)
            plt.title('Reconstructed Image')
            plt.axis('off')
            
            # Best match image
            plt.subplot(1, 3, 3)
            plt.imshow(best_match_image)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
          
        def print_images(reconstructed_image):
         
         
            ax_reconstructed = plt.subplot()
            ax_reconstructed.imshow(np.transpose(reconstructed_image, [1, 2, 0]))
            # ax_reconstructed.set_title("original image") 
            plt.savefig(f'images_created2/{self.idx}.png', dpi=300, bbox_inches='tight')
            plt.close()

        

        if len(self.nodes_values) == len(self.neighbor)+1 :
            self.token = 1
            final_agg = sum_nested_structures_and_negate(self.nodes_array)
           
            gradient =  OrderedDict()

            for key in final_agg:
                gradient[key] = final_agg[key] - self.nodes_array[len(self.nodes_array)-1][key] 
              
                    
            reconstructed = invert_fully_g(gradient["fc1.weight"], gradient["fc1.bias"])

            if reconstructed is not None:
                prob_n = [0,1,2,3]
                for inr in range(0, len(prob_n)):
                    print("sender :",prob_n[inr], "sent ", occurence_num_victim[inr] , "messages to victim")
                
                
                
                for inr in range(0, len(prob_n)):
                    print("node :",prob_n[inr], " learned ",occurence_num_each[inr] , " times")
                    
                for inr2 in range(0, len(prob_n)):
                    print("sender :",self.nodes_values[inr2], "sent ",self.occurence_num_attacker[inr2] , "messages to attacker")
                
             
       

                print_images(reconstructed)

                print("attack done !")
                single_image_path = f'images_created2/{self.idx}.png'
                directory_with_images = 'images_original2'
                compare_images(single_image_path, directory_with_images)
                exit()

                # tal hna
          
        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
          
            return Message(t, self.idx, msg.sender, MessageType.REPLY, (key,))
        return None

# Giaretta et al. 2019
class PassThroughNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync=True):
        r"""Node implementing the pass-through gossiping protocol.
        
        This type of (gossiping) node has been introdued in cite:p:`Giaretta et al. 2019`.
        This pass-through approach consists in making some nodes (in particular hub nodes)
        "bridges" between (low-degree) nodes. This should allow the low-degree nodes to indirectly 
        gossip each other and thus hiding the possible power-law structure of the network. 
        In practice, when node :math:`j` receives a message from :math:`i`, it only performs the 
        merge and update steps with probability :math:`p(i, j) = \min(1, d_i/d_j)` where :math:`d_i`
        and :math:`d_j` are the degrees of :math:`i` and :math:`j`, respectively. Thus, if the 
        sender has lower degree than the receiver, there is a chance the receiver might save 
        the received model as its current model and later propagates it, without going through the
        usual update and merge operations. 

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """
        super(PassThroughNode, self).__init__(idx,
                                              data,
                                              round_len,
                                              model_handler,
                                              p2p_net,
                                              sync)
        self.n_neighs = p2p_net.size(idx)

    # docstr-coverage:inherited
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer, 
                           MessageType.PUSH,
                           (key, self.n_neighs))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, self.n_neighs))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    # docstr-coverage:inherited
    def receive(self, t:int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            
            (recv_model, deg) = msg.value
            recv_model = CACHE.pop(recv_model)
            if  rand() < min(1, deg / self.n_neighs):
                self.model_handler(recv_model, self.data[0])
            else: #PASSTHROUGH
                prev_mode = self.model_handler.mode
                self.model_handler.mode = CreateModelMode.PASS
                self.model_handler(recv_model, self.data[0])
                self.model_handler.mode = prev_mode

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, self.n_neighs))
        return None

# Giaretta et al. 2019
class CacheNeighNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync: bool=True):
        r"""
        As of :class:`PassThroughNode`, this type of (gossiping) node has been introdued in 
        cite:p:`Giaretta et al. 2019`. A :class:`CacheNeighNode` node has as one model slot 
        for each of its neighbours. When receiving a model from a neighbour :math:`j`, instead
        of processing it immediately to update its current model, the node saves it in the 
        corresponding slot. Only when the time to gossip a new model comes, the node picks a 
        random slot and uses the model stored there to perform the MERGE-UPDATE steps.

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """
        super(CacheNeighNode, self).__init__(idx,
                                             data,
                                             round_len,
                                             model_handler,
                                             p2p_net,
                                             sync)
        self.local_cache = {}
    
    # docstr-coverage:inherited
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            if self.local_cache:
                k = random.choice(set(self.local_cache.keys()))
                cached_model = CACHE.pop(self.local_cache[k])
                del self.local_cache[k]
                self.model_handler(cached_model, self.data[0])
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            if self.local_cache:
                k = random.choice(set(self.local_cache.keys()))
                cached_model = CACHE.pop(self.local_cache[k])
                del self.local_cache[k]
                self.model_handler(cached_model, self.data[0])
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key,))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            if sender in self.local_cache:
                CACHE.pop(self.local_cache[sender])
            self.local_cache[sender] = recv_model

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key,))
        return None

# Hegedus 2021
class SamplingBasedNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: SamplingTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync=True):
        super(SamplingBasedNode, self).__init__(idx,
                                                data,
                                                round_len,
                                                model_handler,
                                                p2p_net,
                                                sync)

    # docstr-coverage:inherited          
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, self.model_handler.sample_size))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, self.model_handler.sample_size))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, sample_size = msg.value
            recv_model = CACHE.pop(recv_model)
            sample = TorchModelSampling.sample(sample_size, recv_model.model)
            self.model_handler(recv_model, self.data[0], sample)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, self.model_handler.sample_size))
        return None


# Hegedus 2021
class PartitioningBasedNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: PartitionedTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync=True):
        r"""Standard :class:`GossipNode` with partitioned model.

        This type of node has been first introduced in :cite:p:`Hegedus:2021`.
        The only difference with the standard :class:`GossipNode` is that the model stored
        in the node is partitioned. Thus, both the :meth:`send` and :meth:`receive` methods
        handle the partitioning.

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : PartitionedTMH
            The object that handles the learning/inference of partitioned-based models.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """
        super(PartitioningBasedNode, self).__init__(idx,
                                                    data,
                                                    round_len,
                                                    model_handler,
                                                    p2p_net,
                                                    sync)

    # docstr-coverage:inherited            
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, pid))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, pid))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, pid = msg.value
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0], pid)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, pid))
        return None


# Onoszko 2021
class PENSNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 n_sampled: int=10, #value from the paper
                 m_top: int=2, #value from the paper
                 step1_rounds=200,
                 sync: bool=True):
        """
        TODO :cite:p:`Onoszko:2021`

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        n_sampled : int, default=10
            TODO
        m_top : int, default=2
            TODO
        step1_rounds : int, default=200
            TODO
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        """

        super(PENSNode, self).__init__(idx,
                                       data,
                                       round_len,
                                       model_handler,
                                       p2p_net,
                                       sync)
        assert self.model_handler.mode == CreateModelMode.MERGE_UPDATE, \
               "PENSNode can only be used with MERGE_UPDATE mode."
        self.cache = {}
        self.n_sampled = n_sampled
        self.m_top = m_top
        known_nodes = p2p_net.get_peers(self.idx)
        if not known_nodes:
            known_nodes = list(range(0, self.idx)) + list(range(self.idx + 1, self.p2p_net.size()))
        self.neigh_counter = {i: 0 for i in known_nodes}
        self.selected = {i: 0 for i in known_nodes}
        self.step1_rounds = step1_rounds
        self.step = 1
        self.best_nodes = None
    
    def _select_neighbors(self) -> None:
        self.best_nodes = []
        for i, cnt in self.neigh_counter.items():
            if cnt > self.selected[i] * (self.m_top / self.n_sampled):
                self.best_nodes.append(i)
    
    # docstr-coverage:inherited
    def timed_out(self, t: int) -> int:
        if self.step == 1 and (t // self.round_len) >= self.step1_rounds:
            self.step = 2
            self._select_neighbors()
        return super().timed_out(t)
    
    # docstr-coverage:inherited
    def get_peer(self) -> int:
        if self.step == 1 or not self.best_nodes:
            peer = super().get_peer()
            if peer is None:
                return None
            if self.step == 1:
                self.selected[peer] += 1
            return peer

        return random.choice(self.best_nodes)

    # docstr-coverage:inherited
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol != AntiEntropyProtocol.PUSH:
            LOG.warning("PENSNode only supports PUSH protocol.")

        key = self.model_handler.caching(self.idx)
        return Message(t, self.idx, peer, MessageType.PUSH, (key,))
        
    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0]
        if msg_type != MessageType.PUSH:
            LOG.warning("PENSNode only supports PUSH protocol.")

        if self.step == 1:
            evaluation = CACHE[recv_model].evaluate(self.data[0])
            # TODO: move performance metric as a parameter of the node
            self.cache[sender] = (recv_model, -evaluation["accuracy"]) # keep the last model for the peer 'sender'

            if len(self.cache) >= self.n_sampled:
                top_m = sorted(self.cache, key=lambda key: self.cache[key][1])[:self.m_top]
                recv_models = [CACHE.pop(self.cache[k][0]) for k in top_m]
                self.model_handler(recv_models, self.data[0])
                self.cache = {} # reset the cache
                for i in top_m:
                    self.neigh_counter[i] += 1
        else:
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0])


# Koloskova et al. 2020
class All2AllGossipNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 model_handler: WeightedTMH, #object that handles the model learning/inference
                 p2p_net: P2PNetwork,
                 sync: bool=True):
        r"""
        TODO

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        p2p_net: P2PNetwork
            The peer-to-peer network that provides the list of reachable nodes according to the 
            network topology.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will 
            regularly time out at the same point in the round. If `False`, the node will time out 
            with a fixed delay. 
        use_mh : bool, default=False
            Whether to use the Metropolis-Hastings weighting scheme for the model averaging.
        """
        super(All2AllGossipNode, self).__init__(idx,
                                        data,
                                        round_len,
                                        model_handler,
                                        p2p_net,
                                        sync)
        self.local_cache = {}
    
    # docstr-coverage:inherited
    def timed_out(self, t: int, weights: Iterable[float]) -> int:
        tout = super().timed_out(t)
        if tout and self.local_cache:
            # if not self.use_mh:
            #     weights = [1.0 /(1 + len(self.local_cache))] * (len(self.local_cache) + 1)
            # else:
            #     n = self.p2p_net.size(self.idx)
            #     weights = [1./n] + [1. / (min(self.p2p_net.size(k), n) + 1) for k in self.local_cache]
            self.model_handler([CACHE.pop(k) for k in self.local_cache.values()], self.data[0], weights)
            self.local_cache = {}
        return tout 

    def get_peers(self) -> int:
        return self.p2p_net.get_peers(self.idx)

    # docstr-coverage:inherited
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            return super().send(t, peer, protocol)
        else:
            raise ValueError("All2AllNode only supports PUSH protocol.")

    # docstr-coverage:inherited
    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        varia = 0
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0] if msg.value else None            
        if msg_type == MessageType.PUSH:
            # this should never happen
            if sender in self.local_cache:
                CACHE.pop(self.local_cache[sender])
                # print("me:",self.idx )
                # if self.idx == 72 and sender == 47:
                
                    # print("here are the parameters:", recv_model.state_dict())
                    # self.idx
                    # print("me:",self.idx ,"            sender:", sender, '          model:', recv_model.key[1])
                    # print(weights)
            self.local_cache[sender] = recv_model

        return None