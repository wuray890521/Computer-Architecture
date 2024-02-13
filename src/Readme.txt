分工:
code:311651055
報告:311651052
---------------------------------------------------
一、main.cu:

    1)CPU的運算

    2)呼叫GPU進行運算並回傳時間


二、cuda1、cuda2：分別為way1 & way2的GPU。


　　　　１）cuda1.cu執行方式:請先將main.cu中的include"cuda1.cu"拿掉註解後，註解掉include"cuda2.cu"，運行main.cu。


　　　　２）cuda2.cu執行方式:請先將main.cu中的include"cuda2.cu"拿掉註解後，註解掉include"cuda1.cu"，運行main.cu。


三、輸出結果形式：

input input input input input input input input input input input 
==================================================================
"input的圖片像素矩陣"
==================================================================

GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU 
==================================================================
"GPU output的圖片像素矩陣"
==================================================================

CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU　CPU
==================================================================
"ＣPU output的圖片像素矩陣"
==================================================================　

GPU time =　＂ＧＰＵ回傳的時間＂ｍｓ


