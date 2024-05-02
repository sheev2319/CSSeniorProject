import ast
import decimal
import numpy as np
from stockfish import Stockfish
from itertools import islice

def get_mirror_image(variant_num):
    fen = get_fen(variant_num)
    string = fen[0:8][::-1]
    i = 0
    with open("fen_positions.txt", "r") as file:
        for line in file:
            if string in line:
                return i
            i += 1

decimal.getcontext().prec = 50

arr = [144, 154, 346, 362, 506, 703, 771, 1325, 1543, 1555, 2057, 2476, 2521, 2995, 3016, 3118, 3271, 3627, 4673, 5333, 5514, 5549, 5565, 5745, 5852, 5985, 6783, 9738, 10242, 10899, 10988, 11790, 12141, 12648, 13186, 13384, 13457, 13499, 14133, 14639, 14716, 14731, 14860, 14875, 14978, 15130, 15356, 15383, 16095, 16246, 17722, 17936, 18066, 18082, 18198, 18415, 18539, 18617, 18639, 18693, 19179, 19472, 19880, 20389, 20411, 20506, 22178, 22421, 22870, 23041, 23326, 23397, 23886, 23989, 24165, 24392, 24540, 24542, 24551, 24554, 24659, 25014, 25153, 25319, 27145, 27432, 27471, 27617, 28510, 28513, 29433, 29761, 29992, 30875, 30910, 31022, 31066, 31134, 31172, 31198, 31386, 31469, 32067, 32914, 33348, 33429, 33992, 35151, 35360, 35845, 35885, 36157, 36282, 37292, 38277, 38401, 38422, 38956, 39068, 39102, 39231, 39349, 39481, 40608, 40669, 40679, 41574, 41945, 43980, 44212, 45158, 45533, 46384, 46400, 46650, 47983, 48043, 48288, 48384, 48453, 48947, 49186, 50183, 50410, 50423, 51204, 51607, 52128, 52619, 52626, 52847, 53502, 53532, 53664, 53679, 53804, 54487, 55034, 55357, 55583, 55796, 55875, 55903, 56116, 56167, 56382, 56807, 56974, 57077, 57320, 57667, 57948, 58019, 58687, 59065, 59308, 59315, 59830, 59900, 60598, 60744, 61287, 61657, 61832, 62161, 62246, 62891, 63198, 63771, 63811, 64256, 64358, 65015, 65254, 65521, 65590, 65682, 66184, 66231, 66356, 66764, 67462, 67788, 68277, 68388, 68433, 68495, 68648, 68817, 69060, 69668, 69858, 69928, 69937, 70066, 70184, 70274, 70591, 71095, 71253, 71983, 72102, 72233, 72239, 72299, 72371, 72461, 72495, 72499, 72542, 72689, 72730, 72838, 72843, 72877, 73324, 73405, 73517, 74216, 74258, 74331, 75465, 76112, 77153, 77384, 77569, 78454, 78508, 79002, 79466, 80063, 80265, 80307, 80466, 80475, 80520, 80769, 82619, 82710, 83362, 83968, 84071, 84081, 84388, 84425, 84562, 84656, 84687, 84843, 84868, 84900, 85600, 86738, 87024, 87073, 87369, 88495, 88779, 88844, 88999, 89347, 89613, 90043, 90090, 90099, 90187, 90675, 90928, 91446, 92082, 93002, 94056, 94387, 94675, 95169, 95275, 95574, 95906, 96060, 96167, 96311, 96594, 96820, 99351, 99609, 99874, 100299, 100707, 101112, 101715, 102121, 102238, 102404, 102500, 102599, 103072, 103941, 104400, 104669, 104676, 104935, 106808, 106870, 107002, 107662, 107906, 108056, 108121, 110074, 110419, 110527, 110584, 110650, 110961, 110990, 111483, 111559, 112140, 112355, 112612, 112662, 113265, 113483, 113521, 113661, 113685, 113773, 113822, 113876, 114087, 114273, 114283, 114454, 115850, 117349, 117560, 118272, 118697, 118916, 119577, 119859, 120404, 120610, 121332, 122275, 122518, 123629, 124168, 124438, 124812, 124962, 125083, 125223, 125301, 125368, 125694, 126375, 126566, 126680, 126823, 127581, 127599, 127719, 127819, 127919, 128078, 128098, 128689, 128881, 129169, 129206, 129461, 130524, 130867, 130928, 131124, 131847, 132367, 132438, 132639, 132972, 133118, 133387, 133538, 133922, 133930, 134252, 134256, 134384, 134602, 134675, 134990, 135243, 135316, 135614, 135861, 136255, 136416, 136457, 136599, 136971, 137201, 137355, 137666, 140317, 140588, 140651, 140832, 141063, 141681, 142308, 142333, 142731, 142753, 143026, 143367, 143435, 144046, 144688, 144926, 145742, 146060, 146350, 146364, 147629, 147905, 148438, 148654, 149603, 149642, 149924, 150062, 150574, 150809, 150964, 150989, 152009, 152252, 152446, 152791, 152956, 153811, 153940, 154154, 154661, 155075, 155962, 156477, 157135, 157152, 157173, 157282, 157420, 157465, 158252, 158720, 159245, 161148, 161475, 162714, 163974, 164081, 164515, 164679, 164923, 164925, 165462, 165511, 165933, 166333, 166394, 166890, 167299, 167327, 167516, 167549, 168229, 168336, 168485, 168661, 168956, 169332, 169550, 170079, 170137, 172014, 173396, 173804, 174436, 174758, 175190, 176134, 177013, 177400, 177444, 177545, 177615, 177988, 178147, 178486, 178632, 179129, 179550, 179554, 179676, 180152, 181487, 181499, 182792, 182833, 182843, 183380, 183398, 184065, 184296, 184459, 184762, 185108, 185654, 185735, 186017, 186130, 187843, 188096, 188276, 189394, 189416, 189447, 189604, 190028, 190980, 191741, 191760, 192823, 193144, 193460, 193521, 193540, 194261, 194453, 195296, 195364, 195943, 196552, 197011, 197244, 197425, 197730, 197807, 197917, 198098, 198411, 199299, 199690, 199696, 199988, 200055, 200208, 200361, 201168, 202373, 202417, 202485, 202558, 202612, 202760, 202937, 203263, 203563, 203975, 204587, 205116, 205438, 205515, 206259, 206311, 206570, 206708, 206967, 207376, 207454, 207618, 207992, 208814, 209324, 209806, 209919, 210087, 210536, 210577, 211334, 211458, 211503, 211608, 211877, 212487, 212538, 212547, 212692, 212770, 212800, 212885, 213125, 213323, 213543, 213705, 213998, 215598, 215912, 216053, 216064, 216073, 216197, 216387, 216592, 216612, 216658, 216734, 217045, 217081, 217202, 217774, 218030, 218330, 219134, 219149, 219203, 219232, 219352, 219715, 219783, 221522, 221579, 221583, 221659, 222187, 222508, 223004, 223005, 223190, 223606, 223966, 225004, 225292, 225760, 225763, 226488, 226729, 226854, 227010, 227500, 228063, 228105, 228179, 228192, 228233, 228357, 228423, 228524, 228974, 228982, 229171, 229277, 229451, 229618, 229663, 229779, 229875, 229942, 230058, 230148, 230632, 230633, 231444, 232141, 232659, 232812, 233245, 233421, 233898, 233972, 234216, 234343, 234374, 234460, 234483, 235514, 235566, 235627, 236173, 236244, 236519, 236810, 237314, 237479, 237512, 237529, 237565, 237846, 237947, 237964, 237994, 238054, 238061, 238209, 238311, 238437, 238648, 238711, 238746, 239068, 239178, 239481, 239886, 240020, 240224, 240259, 240310, 240370, 240381, 240451, 240463, 240472, 240473, 240497, 240529, 240598, 240682, 240701, 240710, 240848, 241139, 241454, 242162, 242255, 242344, 242523, 242729, 242804, 242814, 243057, 243402, 244168, 244674, 244808, 244986, 245000, 245032, 246186, 246339, 247146, 247452, 247667, 247924, 248228, 248325, 248392, 248460, 248521, 249191, 249235, 249446, 249699, 249781, 249828, 250204, 250517, 250571, 250967, 251055, 252623, 252804, 252855, 252914, 253072, 253154, 253243, 253537, 253826, 255044, 255453, 256081, 256398, 256816, 256822, 256878, 257265, 257602, 257815, 258168, 258189, 258342, 258383, 258519, 258578, 258967, 259251, 259588, 259807, 260259, 260464, 260594, 260915, 260971, 261201, 261258, 261425, 261552, 261864, 261960, 262181, 262588, 263027, 263432, 263440, 263770, 263908, 265607, 266510, 266619, 266862, 266980, 267822, 270110, 270200, 270228, 270427, 270495, 270518, 270539, 270562, 270674, 270743, 271114, 271236, 271518, 271536, 271607, 271790, 271912, 271930, 272048, 272172, 272224, 272562, 272824, 273242, 273282, 273777, 275015, 275030, 275500, 275576, 275591, 275631, 275658, 275764, 275989, 276822, 276826, 277149, 278265, 278824, 279353, 279714, 279745, 279837, 279901, 280208, 280770, 280947, 281517, 281891, 282011, 282521, 282822, 283049, 283099, 283583, 284204, 284392, 284484, 284635, 284642, 285497, 285713, 285945, 285992, 286261, 286472, 287126, 287578, 287585, 287721, 287758, 287841, 287952, 288488, 288677, 288772, 288795, 290369, 290438, 290771, 291176, 291252, 291407, 291473, 291547, 291755, 291879, 291937, 293922, 294214, 294259, 294480, 294535, 294690, 295563, 296318, 297203, 297346, 297491, 297755, 297832, 297855, 298590, 299354, 299591, 299657, 299716, 299877, 299951, 300254, 300367, 300493, 300514, 301014, 301223, 301483, 302182, 302411, 302425, 302609, 302624, 303059, 303307, 303702, 304417, 304589, 304807, 305750, 306447, 306494, 306796, 307313, 307364, 308808, 309861, 310681, 311311, 311430, 312056, 312365, 312662, 312787, 313017, 313736, 315721, 316937, 317015, 317990, 319087, 319553, 319709, 319937, 320304, 320471, 320540, 320969, 322301, 322793, 323104, 323627, 324587, 325026, 325121, 326542, 326724, 326833, 327013, 327039, 327058, 327218, 329307, 329536, 329707, 329752, 329808, 331407, 331575, 332311, 333733, 335592, 335600, 336354, 336759, 337154, 337473, 338248, 338288, 338730, 338745, 339592, 339795, 341274, 341664, 341745, 342878, 343526, 343671, 343713, 344284, 344289, 345057, 345661, 346201, 346309, 346423, 346654, 348047, 348253, 349152, 349155, 349181, 349484, 349721, 349808, 350311, 350518, 350950, 351099, 351567, 351632, 352518, 353051, 353132, 353483, 353655, 354041, 354042, 354143, 354758, 354797, 354880, 354943, 354983, 355008, 355268, 355292, 355322, 355332, 355375, 355380, 355479, 355483, 355498, 356431, 356948, 357154, 357182, 357366, 357694, 358186, 358417, 358725, 358746, 358800, 358921, 359261, 359826, 359882, 359884, 360280, 362076, 362373, 362525, 363229, 364552, 365214, 365347, 365671, 365842, 366527, 366720, 367643, 367948, 368405, 368563, 368594, 369253, 369454, 369886, 371019, 371425, 371580, 372105, 372308, 372773, 373822, 374749, 375641, 376269, 376365, 376951, 377826, 377913, 378078, 378557, 378686, 380721, 380789, 382808, 383484, 383702, 384340, 384600, 384734, 385071, 385096, 385375, 385935, 386706, 386971, 387049, 387745, 389266, 389312, 389405, 389563, 389949, 390046, 391121, 391544, 391895, 392109, 392126, 392259, 393363, 393820, 394189, 394466, 394487, 394676, 394773, 395116, 396355, 396525, 397425, 397647, 397958, 398128, 398877, 399006, 399739, 399931, 401171, 401172, 402010, 402232, 402451, 403137, 403158, 403452, 403585, 404500, 404600, 405408, 406026, 406119, 406193, 406269, 406294, 406367, 406801, 406924, 406993, 407991, 408042, 408052, 408054, 408599, 408654, 408694, 408708, 408931, 409479, 409540, 409584, 409701, 412396, 412788, 415006, 416174, 417422, 417595, 417652, 417921, 418176, 418972, 419563, 420066, 420143, 420252, 420435, 420508, 420588, 421511, 421865, 422967, 423476, 423625, 423885, 423943, 424026, 424161, 424219, 424664, 425133, 425183, 425223, 425579, 425901, 426265, 427253, 427314, 427629, 428750, 428890, 429229, 429334, 429522, 430227, 430494, 430627, 430758, 431254, 431404, 431527, 431623, 431777, 431842, 432021, 432116, 432276, 432308, 432379, 432402, 432551, 432563, 432566, 432584, 432637, 432697, 432710, 433101, 433405, 433578, 433647, 433680, 433843, 433958, 434112, 434231, 434702, 434961, 435383, 435656, 435731, 436326, 437169, 437418, 437504, 437633, 438021, 438057, 438095, 438269, 438640, 438846, 439930, 440317, 440799, 440900, 441130, 441325, 441463, 441475, 441568, 441602, 441626, 441660, 441769, 442076, 442980, 443097, 443441, 443498, 443621, 445002, 445195, 445437, 445995, 446046, 446258, 446278, 446773, 446894, 447159, 447612, 448211, 448725, 449120, 449612, 449696, 450056, 450472, 451119, 451518, 451884, 451980, 451992, 452084, 452625, 453555, 453753, 453972, 454056, 454215, 454310, 454950, 455910, 456239, 456311, 456366, 456419, 456632, 456634, 456721, 456723, 456757, 456929, 456980, 457056, 457393, 457660, 458084, 458958, 459172, 459584, 459795, 460271, 460707, 460745, 460748, 460903, 460967, 461041, 461240, 461428, 461716, 461914, 461986, 462029, 462069, 462169, 462496, 462527, 462963, 463314, 463987, 464340, 464453, 464846, 464865, 464967, 465141, 465463, 465645, 465692, 466767, 467194, 467596, 468009, 468060, 468336, 468426, 468586, 468654, 469270, 469978, 470035, 470380, 471246, 471565, 472029, 472185, 472975, 474558, 474775, 475332, 475500, 475884, 476243, 476445, 476596, 476723, 476862, 477067, 477537, 478200, 478599, 480043, 480115, 480805, 481217, 482095, 482871, 483146, 483209, 483248, 483314, 483521, 484551, 485566, 486106, 486307, 486339, 487178, 487435, 487511, 487685, 487923, 488197, 488497, 489051, 489269, 489829, 490099, 490125, 490331, 490352, 490966, 491687, 492746, 493005, 493176, 494434, 494498, 494505, 494582, 494812, 495451, 495879, 496459, 496853, 496951, 497004, 497128, 498114, 498984, 500509, 501183, 501743, 501751, 501901, 502024, 502448, 503528, 503645, 504032, 504287, 504465, 504619, 505229, 505417, 506285, 506772, 507066, 507255, 507511, 507517, 507541, 507639, 507775, 510030, 510535, 510810, 511046, 511049, 511053, 511215, 511276, 511289, 511611, 511705, 511749, 511757, 512362, 512376, 512676, 513132, 513903, 513964, 514075, 514381, 514642, 514699, 514888, 515252, 515370, 515422, 515656, 515715, 515742, 515748, 516939, 516980, 517352, 518481, 518621, 518724, 519419, 519950, 520386, 521428, 521507, 522125, 523168, 523211, 524317, 524963, 525019, 525458, 526479, 526918, 527326, 527692, 527969, 528254, 528477, 528780, 528851, 528906, 528976, 529016, 529087, 529133, 529466, 529512, 529686, 530098, 530552, 530608, 530691, 530695, 530957, 531424, 531508, 532012, 532183, 532748, 532896, 533475, 533557, 534003, 535100, 536244, 537106, 538797, 538799, 538805, 539189, 539476, 540050, 540766, 540830, 540893, 541329, 541436, 541476, 541503, 542203, 542865, 543678, 543771, 544804, 545403, 546130, 546446, 546998, 547124, 547659, 548411, 549949, 550167, 550200, 550320, 550654, 550677, 550735, 551099, 551848, 552218, 552886, 552936, 553188, 553380, 553753, 553909, 554169, 554465, 554588, 554614, 556002, 556037, 556964, 557418, 557991, 558294, 559974, 560959, 561688, 561953, 562481, 563311, 563365, 563535, 564210, 564220, 564508, 564878, 565814, 566854, 567162, 567182, 568985, 569274, 569596, 569660, 570146, 570315, 570357, 571167, 572365, 573935, 574142, 574664, 575375, 575469, 575709, 575719, 575803, 576034, 576197, 576274, 576436, 576626, 576657, 576901, 577188, 577360, 577436, 577923, 578009, 578125, 578893, 579675, 579967, 580021, 580074, 580166, 580370, 580376, 581298, 581518, 581569, 581908, 582060, 582201, 582381, 582503, 582585, 583441, 583631, 583984, 584285, 584323, 584587, 584728, 585025, 585113, 585616, 586221, 586595, 586688, 586921, 587311, 587548, 587667, 587771, 588094, 588179, 588554, 588786, 588911, 588925, 588952, 589006, 589048, 589071, 589122, 589124, 589163, 589235, 589262, 589383, 589409, 589769, 589841, 590327, 590361, 590386, 590915, 591199, 591202, 591613, 591691, 591838, 591955, 591987, 592212, 592217, 592600, 592682, 592717, 593240, 593275, 593442, 593786, 594578, 596407, 596754, 597175, 597488, 597555, 597890, 598053, 598407, 598950, 599076, 599224, 599408, 599491, 600942, 600945, 601083, 601366, 601379, 601478, 601787, 601869, 602094, 602221, 602292, 602414, 602813, 605473, 605629, 605702, 605901, 605987, 606132, 606295, 606731, 607037, 607064, 607702, 607897, 608862, 609617, 609924, 609999, 610020, 610577, 610653, 610705, 610776, 611564, 612075, 612892, 612906, 612993, 613223, 613540, 613745, 614229, 614379, 615736, 616906, 617393, 617900, 618692, 618896, 620423, 621173, 621562, 621778, 621921, 621979, 622684, 622722, 623120, 623275, 623742, 624437, 625257, 625287, 625778, 626196, 626577, 626668, 626695, 626798, 626820, 626888, 627705, 628108, 629181, 630060, 630179, 630593, 631144, 631200, 632146, 632294, 632585, 632632, 632716, 632879, 632922, 633138, 633170, 633622, 633752, 633889, 633954, 635743, 636355, 636521, 636872, 637756, 637878, 637960, 638539, 639758, 640406, 640670, 641549, 642753, 643931, 644963, 645026, 645424, 645785, 646030, 646379, 646644, 646690, 646895, 647058, 647263, 647294, 647325, 647536, 647658, 647730, 647755, 647757, 647896, 647900, 647918, 647965, 647980, 648096, 648453, 648652, 648671, 648777, 649142, 649168, 649695, 649831, 649939, 651015, 651371, 651694, 651738, 651747, 651779, 651982, 652720, 652982, 653249, 653385, 653511, 653665, 654047, 654336, 654500, 654620, 654695, 654801, 654880, 654971, 655152, 655213, 655334, 655338, 655590, 655633, 655714, 655868, 656074, 656299, 656484, 656641, 656765, 656837, 656861, 656863, 657323, 657412, 657704, 657810, 657827, 658012, 658443, 659238, 659244, 659310, 659579, 659745, 659750, 659882, 659935, 659982, 659990, 659994, 660715, 660861, 661115, 661176, 661311, 661436, 661688, 661737, 662362, 663712, 663762, 663808, 663862, 664043, 664216, 664356, 664590, 664798, 665039, 665131, 665515, 666532, 666605, 666753, 667249, 667274, 667369, 667691, 667736, 667858, 668700, 668809, 669186, 669586, 669950, 670380, 670660, 670889, 670895, 671128, 671242, 671624, 672065, 672525, 672710, 673118, 673182, 673467, 673552, 673660, 673666, 673791, 673814, 673966, 674087, 674127, 674136, 674276, 674319, 674798, 674869, 674876, 674988, 675494, 675746, 675917, 677377, 677851, 677954, 678650, 678694, 679263, 680887, 681293, 681497, 681503, 681556, 681867, 681975, 682184, 682822, 683130, 683180, 683219, 683768, 683793, 683827, 683942, 684014, 684169, 684334, 684351, 684591, 685066, 685200, 685287, 685383, 685434, 685476, 685633, 685738, 685753, 685826, 685900, 685906, 686339, 686343, 686812, 688253, 688334, 689128, 689207, 689628, 690228, 690249, 691300, 691855, 691971, 691973, 692277, 692297, 692512, 692516, 692630, 692915, 693604, 693855, 693960, 695122, 695440, 695715, 695822, 695995, 696012, 696469, 696608, 696897, 696911, 697356, 697425, 697509, 697565, 697710, 697816, 698072, 698084, 698085, 698201, 698226, 698540, 698671, 698698, 698742, 699877, 700096, 700330, 700815, 701066, 701982, 702273, 703641, 703665, 704493, 704588, 705054, 705073, 705090, 705554, 706084, 706185, 706253, 707453, 707863, 708955, 709413, 709430, 709539, 709575, 709705, 709762, 709833, 709937, 710095, 710211, 710215, 710296, 710475, 710629, 710647, 710755, 710936, 711934, 712544, 713020, 714641, 714998, 715140, 715311, 715350, 715451, 716339, 716508, 716605, 716726, 717145, 717791, 718003, 718796, 719159, 719217, 719265, 719569, 719696, 719705, 719730, 719748]
arr_as_set = set(arr)
num_threefold_repetitions_per_variant = [6, 2, 3, 3, 4, 0, 1, 7, 0, 1, 0, 0, 1, 1, 2, 1, 2, 4, 1, 6, 3, 2, 0, 2, 8, 2, 1, 3, 0, 2, 2, 4, 7, 3, 0, 0, 4, 0, 2, 3, 0, 9, 1, 1, 2, 1, 1, 3, 2, 1, 0, 4, 5, 0, 3, 2, 0, 0, 2, 0, 2, 2, 1, 1, 4, 2, 1, 2, 2, 1, 3, 5, 1, 2, 6, 3, 3, 2, 2, 4, 2, 1, 4, 1, 1, 4, 1, 4, 3, 2, 1, 6, 2, 6, 2, 2, 11, 5, 2, 2, 1, 1, 1, 2, 2, 2, 1, 6, 0, 0, 2, 2, 7, 3, 1, 1, 3, 1, 3, 2, 5, 2, 1, 0, 1, 2, 2, 3, 4, 1, 0, 0, 2, 2, 2, 1, 5, 1, 1, 4, 0, 0, 3, 2, 2, 0, 1, 6, 2, 2, 2, 8, 4, 0, 1, 0, 2, 1, 2, 2, 2, 1, 0, 2, 1, 2, 4, 3, 3, 1, 7, 2, 3, 0, 4, 1, 3, 3, 3, 6, 3, 4, 3, 2, 0, 0, 0, 4, 2, 2, 3, 2, 2, 1, 2, 2, 1, 2, 1, 3, 2, 3, 1, 4, 0, 3, 2, 1, 1, 6, 0, 2, 1, 0, 1, 1, 1, 0, 2, 4, 2, 3, 1, 4, 4, 2, 3, 0, 0, 1, 0, 2, 1, 2, 1, 0, 5, 3, 2, 3, 1, 2, 0, 3, 2, 3, 2, 2, 2, 0, 2, 1, 4, 1, 1, 2, 0, 3, 2, 2, 2, 1, 3, 4, 2, 1, 5, 1, 1, 3, 4, 3, 1, 2, 1, 5, 3, 1, 1, 3, 3, 2, 3, 7, 4, 1, 0, 2, 9, 3, 2, 1, 6, 1, 0, 4, 2, 3, 2, 0, 2, 3, 3, 1, 8, 5, 7, 2, 1, 1, 3, 3, 5, 0, 5, 2, 5, 9, 5, 2, 16, 3, 1, 6, 2, 1, 5, 0, 2, 2, 3, 4, 4, 3, 4, 0, 1, 6, 2, 0, 2, 2, 3, 3, 6, 2, 2, 5, 4, 3, 2, 4, 0, 0, 1, 4, 0, 1, 0, 0, 10, 2, 9, 2, 2, 1, 2, 7, 0, 3, 0, 2, 3, 3, 2, 2, 2, 3, 2, 4, 2, 4, 1, 6, 2, 2, 0, 3, 5, 3, 0, 1, 5, 0, 1, 1, 3, 3, 1, 6, 4, 3, 1, 4, 3, 1, 2, 1, 2, 3, 0, 1, 0, 1, 1, 2, 3, 2, 1, 0, 1, 0, 2, 1, 0, 1, 3, 4, 0, 1, 2, 1, 1, 2, 0, 3, 4, 0, 0, 5, 0, 1, 1, 1, 1, 0, 0, 2, 1, 3, 1, 3, 1, 1, 0, 3, 0, 1, 3, 2, 2, 3, 1, 0, 2, 4, 2, 3, 3, 0, 3, 2, 3, 15, 0, 2, 4, 2, 4, 4, 1, 0, 1, 2, 1, 0, 2, 3, 2, 0, 2, 3, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 3, 0, 0, 2, 0, 0, 1, 2, 3, 3, 1, 2, 2, 0, 0, 5, 1, 1, 4, 1, 1, 4, 3, 0, 2, 1, 3, 1, 2, 1, 2, 0, 3, 3, 1, 2, 1, 6, 3, 1, 7, 2, 3, 0, 0, 1, 1, 0, 0, 1, 1, 0, 3, 2, 1, 1, 6, 0, 2, 1, 2, 5, 4, 2, 1, 2, 1, 2, 3, 2, 2, 6, 13, 2, 7, 2, 3, 1, 1, 3, 5, 1, 1, 3, 8, 2, 2, 3, 0, 3, 2, 4, 2, 1, 2, 2, 2, 1, 5, 1, 1, 5, 1, 1, 8, 5, 2, 1, 2, 2, 7, 4, 5, 2, 1, 5, 4, 0, 2, 1, 6, 1, 2, 1, 2, 2, 1, 0, 1, 2, 3, 4, 2, 1, 1, 0, 2, 2, 1, 1, 5, 0, 1, 1, 3, 2, 4, 1, 2, 5, 1, 1, 1, 2, 0, 5, 1, 2, 4, 0, 1, 1, 0, 1, 2, 4, 0, 2, 4, 2, 0, 2, 6, 1, 0, 0, 2, 7, 4, 3, 1, 4, 3, 7, 0, 3, 0, 3, 1, 2, 0, 2, 1, 2, 0, 2, 2, 1, 1, 3, 2, 8, 3, 5, 2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 4, 1, 1, 6, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 4, 4, 1, 1, 4, 3, 3, 0, 2, 1, 2, 1, 0, 1, 1, 1, 2, 0, 3, 3, 1, 1, 1, 2, 0, 1, 3, 3, 1, 0, 1, 0, 2, 1, 5, 6, 4, 3, 1, 1, 6, 0, 4, 5, 1, 2, 4, 3, 1, 3, 4, 3, 14, 2, 4, 4, 5, 4, 3, 1, 0, 0, 2, 3, 3, 3, 2, 0, 6, 4, 3, 0, 0, 0, 5, 3, 2, 2, 1, 1, 3, 4, 1, 1, 4, 3, 1, 1, 0, 1, 2, 1, 1, 0, 1, 2, 3, 3, 2, 1, 2, 2, 6, 1, 1, 1, 0, 3, 2, 1, 6, 3, 3, 0, 1, 2, 1, 3, 1, 0, 2, 1, 1, 0, 1, 1, 1, 2, 3, 4, 13, 4, 3, 3, 0, 5, 2, 3, 3, 5, 7, 5, 7, 3, 4, 2, 9, 1, 5, 2, 1, 1, 6, 4, 1, 2, 4, 3, 2, 2, 2, 5, 1, 3, 3, 9, 6, 2, 1, 0, 3, 2, 1, 0, 1, 4, 3, 4, 4, 5, 6, 6, 3, 0, 2, 2, 1, 2, 1, 3, 6, 1, 2, 1, 4, 3, 4, 9, 4, 0, 3, 2, 1, 1, 0, 2, 2, 4, 3, 0, 2, 0, 3, 9, 6, 0, 1, 2, 0, 1, 5, 0, 4, 1, 2, 3, 6]

def read_game(line):
    data = line[:-1].split("_")
    position_number = int(data[0])
    entropy = ast.literal_eval(data[1])
    transition_probabilities = ast.literal_eval(data[2])
    for prob in transition_probabilities:
        assert(prob<=1)
    moves_taken = ast.literal_eval(data[3])
    evaluations = ast.literal_eval(data[4])
    outcome = int(data[5])
    return [position_number,entropy,transition_probabilities,moves_taken,evaluations,outcome]

def print_all_with_threefold_repetition(file_path):
    stockfish = Stockfish(parameters={"UCI_Chess960": True})
    repeated_positions = []
    num_repeated_positions_per_variant = [0 for i in range(960)]
    with open(file_path, 'r') as file:
        game_num=0
        for line in file:
            print(game_num)
            data = read_game(line)
            variant_num = data[0]
            moves_taken = data[3]
            if check_threefold_repetition(get_fen(variant_num), moves_taken, stockfish=stockfish):
                repeated_positions.append(game_num)
                num_repeated_positions_per_variant[variant_num] += 1
            game_num += 1
    print(repeated_positions)
    print(num_repeated_positions_per_variant)

def read_file(file_path):
    ply_entropies = []
    outcomes = []
    ply40_negative_log_prob = []
    with open(file_path, 'r') as file:
        sequence_probabilities = [[] for _ in range(40)] # length 40, for each index i, have an array tracking sequence probability through ply i
        overall_moves_taken = []
        overall_transition_probabilities = []
        variant_entropies = []
        num_whitewins = 0
        num_draws = 0
        num_blackwins = 0
        variant_num = 0
        game_num = 0
        for line in file:
            print(game_num)
            data = read_game(line)
            #entropies = data[1]
            transition_probabilities = data[2]
            outcome = data[5]
            moves_taken = data[3]
            #position_num = data[0]
            if game_num in arr_as_set:
                num_draws += 1
                if game_num % 750 == 749 and game_num >= 1:
                    # we are about to enter a new variant        
                    # log results of the old variant                 
                    for ply in range(40):
                        negative_logs = [-decimal.Decimal(prob).ln() for prob in sequence_probabilities[ply]]
                        variant_entropies.append(sum(negative_logs)/len(negative_logs))
                    ply40_negative_log_prob.append(negative_logs)
                    #ply40_entropy_sds.append(np.std(negative_logs))
                    ply_entropies.append(variant_entropies)
                    outcomes.append((num_whitewins, num_draws, num_blackwins))                 
                    # reset/update vars
                    sequence_probabilities = [[] for _ in range(40)]
                    variant_entropies = []
                    variant_num += 1
                    num_whitewins = 0
                    num_draws = 0
                    num_blackwins = 0
                game_num += 1
                continue
            
            prod = decimal.Decimal(1)
            l = len(transition_probabilities)
            for i in range(40):
                if l <= i:
                    break
                prod *= decimal.Decimal(transition_probabilities[i])
                sequence_probabilities[i].append(prod)
            overall_moves_taken.append(moves_taken)
            overall_transition_probabilities.append(transition_probabilities)

            if outcome == 1:
                num_whitewins += 1
            elif outcome == 0:
                num_draws += 1
            else:
                num_blackwins += 1
                
            if game_num % 750 == 749 and game_num >= 1:
                # we are about to enter a new variant
                
                # log results of the old variant
                for ply in range(40):
                    negative_logs = [-decimal.Decimal(prob).ln() for prob in sequence_probabilities[ply]]
                    variant_entropies.append(sum(negative_logs)/len(negative_logs))
                ply40_negative_log_prob.append(negative_logs)
                ply_entropies.append(variant_entropies)
                   
                outcomes.append((num_whitewins, num_draws, num_blackwins))
                
                # reset/update vars
                sequence_probabilities = [[] for _ in range(40)]
                variant_entropies = []
                variant_num += 1
                num_whitewins = 0
                num_draws = 0
                num_blackwins = 0
            game_num += 1
    return (ply_entropies, outcomes, ply40_negative_log_prob)

def get_fen(position_num):
    with open("fen_positions.txt", "r") as file:
        line = next(islice(file, position_num, None), None)
        return line

def check_threefold_repetition(fen, moves, stockfish):
    # Initialize the board with the FEN string
    stockfish.set_fen_position(fen)
    
    state_count = {}

    # Apply each move and check board state repetitions
    for move in moves:
        stockfish.make_moves_from_current_position([move])
        current_fen = stockfish.get_fen_position()
        current_fen = " ".join(current_fen.split(" ")[:-2])
        # Track the occurrence of each board state
        if current_fen in state_count:
            state_count[current_fen] += 1
        else:
            state_count[current_fen] = 1
        # Check for threefold repetition
        if state_count[current_fen] == 3:
            return True
    return False

def get_move_sequence_probability(transition_probabilities, ply_num=39, prev_ply_prob=None):
    if len(transition_probabilities) <= ply_num:
        return None
    if prev_ply_prob != None:
        assert(prev_ply_prob <= 1)
        # if the have the probability of the previous ply, we just use that
        # multiply that probability by the probability of the most recent move
        return decimal.Decimal(prev_ply_prob) * decimal.Decimal(transition_probabilities[ply_num])
    prod = decimal.Decimal(1)
    for i in range(ply_num):		
        prod *= decimal.Decimal(transition_probabilities[i])
    return prod

def sequence_entropy(sequence_probabilities):
    """Take as input an array containing the probabilities of the various games,
    and calculate the entropy of the game through ply t """
    #s = decimal.Decimal(sum(sequence_probabilities))
    v = decimal.Decimal(0)
    for prob in sequence_probabilities:
        prob_decimal = decimal.Decimal(prob)  # Convert probability to Decimal
        v -= (prob_decimal * prob_decimal.ln())  # Calculate logarithm using ln() method of Decimal
    return v

def get_avg_candidate_moves(sequence_probabilities, curr_variant_state_entropies):
    """Sequence probabilities: length 40 array where entry i is an array of sequence probabilites through ply i
    curr_variant_state_entropies: length 40 array where entry i is an array of ply i entropies """
    avg_candidate_moves = []
    for i in range(40):
        s = sum(sequence_probabilities[i])
        v = decimal.Decimal(0)
        for j in range(len(sequence_probabilities[i])):
            prob_decimal = decimal.Decimal(sequence_probabilities[i][j])
            v += (prob_decimal * (decimal.Decimal(np.exp(1)) ** decimal.Decimal(curr_variant_state_entropies[i][j])))
        avg_candidate_moves.append(v/s)
    return avg_candidate_moves

def get_WDL_string(outcome):
        white_win = outcome[0]
        draw = outcome[1]
        black_win = outcome[2]
        s = white_win+draw+black_win
        return f"White win: {100 * white_win/(s):.0f}%,  Draw: {100 * draw/(s):.0f}%,  Black win: {100 * black_win/(s):.0f}%"

def write_results_save_arrays():
    ply_entropies, outcomes, ply40_negative_log_prob = read_file("filtered_data.txt")
    ply_entropies = np.array(ply_entropies) 
    outcomes = np.array(outcomes)
        
    np.savez('entropies_and_outcomes.npz', arr1=ply_entropies, arr2=outcomes)
    with open("final_results2.txt", "w") as file:
        for i in range(960):
            file.writelines(f"Variant number: {i},  Outcomes: {get_WDL_string(outcomes[i])},  Move 5 Entropy: {float(ply_entropies[i][9]):.1f},  Move 10 Entropy: {float(ply_entropies[i][19]):.1f},  Move 20 Entropy: {float(ply_entropies[i][39]):.1f}\n")
    
    with open('ply40_negative_log_prob.txt', "w") as file:
        for subarray in ply40_negative_log_prob:
            # Convert each Decimal to a float and format it to two decimal places
            formatted_subarray = [f"{float(val):.2f}" for val in subarray]
            # Join the formatted strings with commas and write to the file
            file.write(','.join(formatted_subarray) + '\n')
            
#write_results_save_arrays()
