#antialias graph

import getsdata

DIM = getsdata.DIM
TRAIN_DIR = "/home/banditcat/tfmuse/train/"
TRAIN_FILE = "t"
STOCK_DIR = getsdata.STOCK_DIR 
PATCH_SIZE = 5
L1_FEATURES = 32
L2_FEATURES = 64
DENSE_FEATURES = 1024
BATCH_SIZE = 50
STEPS = 15

NUM_STOCKS = 3
GOOD_TICKER_THRESHHOLD = 0
BAD_TICKER_THRESHHOLD = -1
POINTS_CAP = 0
VOLUME_PRICE_MINIMUM = 20000 / NUM_STOCKS


import numpy
import math
import datetime
import calendar
import tensorflow as tf
import matplotlib.pyplot as mpl
mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['interactive'] = 'True'



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [ 1, 1, 1, 1 ], padding = 'SAME' )

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize = [ 1, 2, 2, 1 ],
                        strides = [ 1, 2, 2, 1 ], padding = 'SAME' )

#vars
x = tf.placeholder( tf.float32, [ None, DIM * DIM ] )           
y_ = tf.placeholder( tf.float32, [ None, DIM ] )
lasts = tf.placeholder( tf.int64, [ None ] )
prices = tf.placeholder( tf.float32, [ None ] )
volumes = tf.placeholder( tf.float32, [ None ] ) 

#Conv 1
W_conv1 = weight_variable( [ PATCH_SIZE, PATCH_SIZE, 1, L1_FEATURES ] )
b_conv1 = bias_variable( [ L1_FEATURES  ] )
x_image = tf.reshape(x, [ -1, DIM, DIM, 1 ])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Conv 2
W_conv2 = weight_variable( [ PATCH_SIZE, PATCH_SIZE, L1_FEATURES, L2_FEATURES ])
b_conv2 = bias_variable( [ L2_FEATURES ] )
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Dense layer
W_fc1 = weight_variable( [ ( DIM // 4 ) *  ( DIM // 4 )  * L2_FEATURES, DENSE_FEATURES ] )
b_fc1 = bias_variable( [ DENSE_FEATURES ] )
h_pool2_flat = tf.reshape( h_pool2, [-1, ( DIM // 4 ) *  ( DIM // 4 )  * L2_FEATURES ] )
h_fc1 = tf.nn.relu( tf.matmul( h_pool2_flat, W_fc1 ) + b_fc1 )

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Softmax
W_fc2 = weight_variable( [ DENSE_FEATURES, DIM ] )
b_fc2 = bias_variable( [ DIM ] )
y_conv = tf.nn.softmax( tf.matmul( h_fc1_drop, W_fc2 ) + b_fc2 )


#model
cross_entropy = -tf.reduce_sum( y_ * tf.log( y_conv ) )
train_step = tf.train.AdamOptimizer( 1e-4 ).minimize( cross_entropy )
score = tf.argmax( y_conv, 1 )
actual_score = tf.argmax( y_, 1 )
correct_prediction = tf.equal( score, actual_score )
accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )
predictions = score - lasts
actuals = actual_score - lasts
growth_bool = tf.greater( predictions, tf.cast( tf.fill( tf.shape( predictions ), GOOD_TICKER_THRESHHOLD ), tf.int64 ) )
shrink_bool = tf.less( predictions, tf.cast( tf.fill( tf.shape( predictions ), BAD_TICKER_THRESHHOLD ), tf.int64 ) )
vol_price_bool = tf.greater( volumes * prices, VOLUME_PRICE_MINIMUM )
growth_combined_bool = tf.cast( tf.cast( growth_bool, tf.float32 ) * tf.cast( vol_price_bool, tf.float32 ), tf.bool )
growth_tickers = tf.reshape( tf.where( growth_combined_bool ), [ -1 ] )
shrink_combined_bool = tf.cast( tf.cast( shrink_bool, tf.float32 ) * tf.cast( vol_price_bool, tf.float32 ), tf.bool )
shrink_tickers = tf.reshape( tf.where( shrink_combined_bool ), [ -1 ] )


def get_imc1():
  return tf.reshape( W_conv1 + b_conv1, [ -1, L1_FEATURES * PATCH_SIZE ] )
def get_imc2():
  return tf.reshape( W_conv2 + b_conv2, [ -1, L2_FEATURES * PATCH_SIZE ] )
def get_imdl():
  return tf.reshape( W_fc1 + b_fc1, [-1, DENSE_FEATURES ] )
def get_imsm():
  return tf.reshape( W_fc2 + b_fc2, [-1, L1_FEATURES ] )

saver = tf.train.Saver()

  
sess = tf.Session()

chkpt = tf.train.latest_checkpoint( TRAIN_DIR ) 
if chkpt != None:
  print( "Loading", chkpt )
  saver.restore( sess, chkpt )
else:
  mnist = getsdata.read_data_sets( STOCK_DIR )
  sess.run( tf.initialize_all_variables() )
  for i in range( STEPS * 1000 + 1 ):
    batch = mnist.train.next_batch( BATCH_SIZE )
    sess.run( train_step, feed_dict = { x: batch[ 0 ], y_: batch[ 1 ], keep_prob: 0.25 } )
    if i % 100 == 0:
      train_accuracy = accuracy.eval( session=sess, feed_dict = {
        x:batch[0], y_: batch[1], keep_prob: 1.0})
      print( "step %d, training accuracy %g epoch %d" % ( i, train_accuracy, mnist.train.epochs_completed ) )
    if i % 1000 == 0:
      print( "Test accuracy %g" % accuracy.eval( session = sess, feed_dict = {
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0 } ) )
      saver.save( sess, TRAIN_DIR + TRAIN_FILE, global_step = i )

      

imc1 = get_imc1()
imc2 = get_imc2()
imdl = get_imdl()
imsm = get_imsm()        

mpl.figure( 1, figsize = ( 5, 5 ), dpi = 129.582774711 )
mpl.subplot( 411 )
mpl.imshow( imc1.eval(session=sess), interpolation='nearest', cmap="prism")
mpl.axis("off")
mpl.subplot( 412 )
mpl.imshow( imc2.eval(session=sess), interpolation='nearest', cmap="prism")
mpl.axis("off")
mpl.subplot( 413 )
mpl.imshow( imdl.eval(session=sess), interpolation='nearest', cmap="prism")
mpl.axis("off")
mpl.subplot( 414 )
mpl.imshow( imsm.eval(session=sess), interpolation='nearest', cmap="prism")
mpl.axis("off")
mpl.draw()
mpl.show()


foog, _, _, _, _, _ = getsdata.build_predictions( STOCK_DIR, 2016, 1, 1 )
mpl.figure( 2, figsize = ( 5, 5 ), dpi = 129.582774711 )
foog = foog.reshape( -1, DIM, DIM )[ 0:10 ]
foog = foog.reshape( -1, DIM )
mpl.imshow( foog, interpolation='nearest', cmap="hot")
mpl.axis( "off" )
mpl.draw()
mpl.show()


feb = []
tot = 100.0
month = 1
year = 2015
total_perfs = []

##  Do the prediction
# money = 1000000.0
# timgs, tticks, tlsts = getsdata.build_today()
# print( len( timgs ), "SAdasd" )
# timgs = timgs.reshape( -1, DIM * DIM )

# ttgp = []
# tgprds = growth_tickers.eval( session = sess, feed_dict = { x: timgs, lasts: tlsts, keep_prob: 1.0 } )
# # Remove low-volume
# for i in tgprds:
#   vol, _ = getsdata.get_volume_and_price_for_today( tticks[ i ] + "\n" )
#   if vol >= 10000:
#     ttgp.append( i )
# tgprds = ttgp

# tticks = numpy.array( tticks )
# cnt = len( tgprds )
# if len( tgprds ) > 0:
#   for i in tgprds:
#     vol, prc = getsdata.get_volume_and_price_for_today( tticks[ i ] + "\n" )
#     tospend = money / cnt
#     cnt -= 1
#     to_buy = round( tospend / prc )
#     money -= to_buy * prc
#     if money < 0:
#       money += prc
#       to_buy -= 1
#     print( tticks[ i ], " Volume:", vol, " Price:", prc, "To buy:", to_buy, "Money left:", money )
# else:
#   print( "None!", money )

# Verify
for year in range( 2015, 2017 ):
  for month in range( 1, 13 ):
    monthtot = 1.0
    for day in range( 1, 33 ):
      if year < 2016 or month < 4: 
        foof, fool, ticks, lsts, prcs, vols = getsdata.build_predictions( STOCK_DIR, year, month, day )
        if len( ticks ) != 0:
          foof = foof.reshape( -1, DIM * DIM )
          ticks = numpy.array( ticks )
          gprds = growth_tickers.eval( session = sess, feed_dict = { x: foof, lasts: lsts, keep_prob: 1.0,
                                                                   prices: prcs, volumes: vols } )

          #Added this to select number of stocks
          prds = predictions.eval( session = sess, feed_dict = { x: foof, lasts: lsts, keep_prob: 1.0 } )
          evls = actuals.eval( session = sess, feed_dict = { x: foof, y_: fool, lasts: lsts, keep_prob: 1.0 } )
          numpy.random.shuffle( gprds )
          gprds = numpy.intersect1d( numpy.argsort( prds ), gprds )[ -NUM_STOCKS : ]
        
        
          #numpy.random.shuffle( gprds )
          #gprds.sort( kind = "mergesort" )
          #gprds = gprds[ 0 : 5 ]
        
          sprds = shrink_tickers.eval( session = sess, feed_dict = { x: foof, lasts: lsts, keep_prob: 1.0,
                                                                     prices: prcs, volumes: vols } )
          #gprds = numpy.random.choice( numpy.arange( 0, len( ticks ) - 1 ), len( ticks ) // 100, replace = False )
          wkdy = datetime.date( year, month, day ).weekday()
          print( calendar.day_name[ wkdy ], year, month, day )
          if len( gprds ) != 0:
            prfs = []
            nprds = []
            for i in gprds:
              name = ticks[ i ]
              prf, prc, vol = getsdata.get_profit( ticks[ i ] + "\n", STOCK_DIR, year, month, day )
              #if prf != 1.0:
              print( name, prf, "\t Price", prc, "\t Volume", vol, "\t Predicted points:", prds[ i ], "\t Actual points:", evls[ i ] )
              prfs.append( prf )
              nprds.append( prds[ i ] )
            prfs = numpy.array( prfs )
            #BUGBUG weighting gives bad resukts
            nprds = numpy.array( nprds )
            nprds = numpy.clip( nprds, -1000000000.0, POINTS_CAP )
            if POINTS_CAP <= 0:
              nprds = numpy.ones( len( nprds ) )
            nprdss = nprds.sum()
            nprds = numpy.array( [ x / nprdss for x in nprds ] )
            tprf = ( prfs * nprds ).sum()
            total_perfs.append( tprf )
            print( tprf )
            tot *= tprf
            monthtot *= tprf
            print( tot )
            print()
          else:
            print( "====================================== NONE =======================================" )
    print( "Month total", monthtot )

total_perfs = numpy.array( total_perfs )

total_avg = numpy.average( total_perfs )
total_var = numpy.var( total_perfs )
total_std = numpy.std( total_perfs )
num_days = 349
total_buying_days = len( total_perfs )
daily_rate = ( tot / 100 ) ** ( 1 / num_days )
three_monthly_rate = daily_rate ** ( 30.42 * 3 )

print( "\nExpected return:", tot, "percent." 
       "\nTotal days trading:", total_buying_days,
       "\nOverall average stock performance:", total_avg,
       "\nDaily rate:", daily_rate,
       "\nThree month rate:", three_monthly_rate,
       "\nOverall performance variance, standard deviation:", total_var, ",", total_std,
       "\nSharpe ratio over three months assuming 0.30% t-bill rate:", ( three_monthly_rate - 1.003 ) / total_std ) 
sess.close()


