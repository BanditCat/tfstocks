# Copyright 2016 Jonathan DuBois

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import os.path
import urllib.request
import numpy
import math
import datetime
class Now( object ):
  pass
now = Now()
now.year = 2016
now.month = 3
now.day = 10

YEAR_TO_GET_FROM = 2002
STOCK_DIR = "/home/banditcat/tfmuse/sdata/"
STOCK_TODAY_DIR = STOCK_DIR + "today/"
DIM = 64

def filtertickers( tc ):
  if len( tc ) >= 5 and ( tc[ 4 ] == "W" or tc[ 4 ] == "R" or tc[ 4 ] == "U" or
                          tc[ 4 ] == "L" or tc[ 4 ] == "Z" or tc[ 4 ] == "C" or
                          tc[ 4 ] == "V" or tc[ 4 ] == "A" or tc[ 4 ] == "M" or
                          tc[ 4 ] == "P" ):
    return False
  else:
    return True
      
def get_tickers( tickers, path, year ):
  """Get the ticker files"""
  for tc in tickers:
    #Take off newline
    tc = tc[:-1]
    fn = path + tc + ".txt"
    #BUGBUG make date flexible
    urlf = 'http://real-chart.finance.yahoo.com/table.csv?s=' + tc + '&a=10&b=1&c=2015&d=02&e=11&f=2016&g=d&ignore=.csv'
    if not os.path.isfile( fn ):
      print( "Fetching", urlf )
      _, _ = urllib.request.urlretrieve( urlf, fn )

      
def filter_date( l ):
  year = int( l[ 0 : 4 ] )
  day = int( l[ 5 : 7 ] )
  month = int( l[ 8 : 10 ] )
  if year < 2015 and year >= 2004:
    return True
  else:
    return False

def get_adjusted_price( ln ):
  items = ln.split( "," )
  return ( float( items[ 1 ] ) + float( items[ 6 ] ) ) / 2.0

def get_high( ln ):
  items = ln.split( "," )
  return float( items[ 2 ] )

def get_volume( ln ):
  items = ln.split( "," )
  return float( items[ 5 ] )

def get_closing_price( ln ):
  items = ln.split( "," )
  return float( items[ 4 ] )

def get_opening_price( ln ):
  items = ln.split( "," )
  return float( items[ 1 ] )

def get_low( ln ):
  items = ln.split( "," )
  return float( items[ 3 ] )


def compare_date( lns, i, year, month, day ):
  cy = int( lns[ i ][ 0 : 4 ] )
  cm = int( lns[ i ][ 5 : 7 ] )
  cd = int( lns[ i ][ 8 : 10 ] )
  if year > cy:
    return 1
  elif year < cy:
    return -1
  elif month > cm:
    return 1
  elif month < cm:
    return -1
  elif day > cd:
    return 1
  elif day < cd:
    return -1
  else:
    return 0
  
def find_date( lns, year, month, day ):
  first = 0
  last = len( lns ) - 1
  
  while first <= last:
    midpoint = ( first + last ) // 2
    cmp = compare_date( lns, midpoint, year, month, day )
    if cmp == 0:
      return midpoint
    else:
      if cmp < 0:
        last = midpoint - 1
      else:
        first = midpoint + 1
        
  return None  

  

def get_profit( ticker, path, year, month, day ):
  #BUGBUG fudge 
  tc = load_ticker( ticker, path )
  i = find_date( tc, year, month, day )
  if i:
    buy_price = get_closing_price( tc[ i - 1 ] )
    price = get_opening_price( tc[ i ] )
    vol = get_volume( tc[ i - 1 ] )
    mult = price / buy_price 
    if mult < 3.0:
      return mult, buy_price, vol
    else:
      return 1.0, buy_price, vol
    
def get_volume_and_price_for_today( ticker ):
  tc = load_ticker( ticker, STOCK_TODAY_DIR )
  i = find_date( tc, now.year, now.month, now.day )
  if i:
    vol = get_volume( tc[ i ] )
    prc = get_closing_price( tc[ i ] )
  return vol, prc
  
  
def build_image( lns, ia ):
  i = ia + DIM // 2
  img = []
  avg = get_adjusted_price( lns[ i ] )
  for j in range( DIM // 2 ):
    apl = get_low( lns[ i + j ] )
    aph = get_high( lns[ i + j ] )
    apa = get_opening_price( lns[ i + j ] )
    apa2 = get_closing_price( lns[ i + j ] )
    pxav = ( apa / avg ) * ( ( DIM - 1 ) / 2 )
    pxa = int( round( pxav) )
    pxa = min( pxa, ( DIM - 1 ) )           
    pxav2 = ( apa2 / avg ) * ( ( DIM - 1 ) / 2 )
    pxa2 = int( round( pxav2 ) )
    pxa2 = min( pxa2, ( DIM - 1 ) )           
    pxl = int( round( ( apl / avg ) * ( ( DIM - 1 ) / 2 ) ) )
    pxl = min( pxl, ( DIM - 1 ) )           
    pxh = int( round( ( aph / avg ) * ( ( DIM - 1 ) / 2 ) ) )
    pxh = min( pxh, ( DIM - 1 ) )           

    col = numpy.zeros( DIM )
    for px in range( pxl, pxh + 1 ):
      col[ px ] = 0.1
    col[ pxa ] = 1.0
    img.append( col )

    col = numpy.zeros( DIM )
    for px in range( pxl, pxh + 1 ):
      col[ px ] = 0.1
    col[ pxa2 ] = 1.0
    img.append( col )
    #BUGBUG
  return ( img, avg, pxa2 )

def build_label( lns, i, avg ):
  lbl = numpy.zeros( DIM )
  apa = get_opening_price( lns[ i ] )
  pxav = ( apa / avg ) * ( ( DIM - 1 ) / 2 )
  pxa = int( round( pxav) )
  pxa = min( pxa, ( DIM - 1 ) )           
  pxaf = pxav - pxa
  apl = get_low( lns[ i ] )
  pxl = int( round( ( apl / avg ) * ( ( DIM - 1 ) / 2 ) ) )
  pxl = min( pxl, ( DIM - 1 ) )
  aph = get_high( lns[ i ] )
  pxh = int( round( ( aph / avg ) * ( ( DIM - 1 ) / 2 ) ) )
  pxh = min( pxh, ( DIM - 1 ) )
  for px in range( pxl, pxh + 1 ):
    lbl[ px ] = 0.1

  lbl[ pxa ] = 1.0

  return lbl
  

def load_ticker( ticker, path ):
  tc = ticker[:-1]
  with open( path + tc + ".txt" ) as f:
    lns = f.readlines()
  lns = lns[1:]
  lns.reverse()
  return lns


def load_tickers( path, year ):
  with open( path + "ticksraw.txt" ) as f:
    tickers = f.readlines()
    
  tickers = list( filter( filtertickers, tickers ) )  
  get_tickers( tickers, path, year )
  return tickers



def build_chunks( path ):
  tickers = load_tickers( path, YEAR_TO_GET_FROM )

  imgs = []
  lbls = []

  for tc in tickers:
    lns = load_ticker( tc, path )
    lns = list( filter( filter_date, lns ) )

    i = 0
    while i + DIM < len( lns ):
      i += random.randint( 10, DIM )
      if i + DIM + 1 < len( lns ):
        img, avg, _ = build_image( lns, i )
        lbl = build_label( lns, i + DIM, avg )
        imgs.append( img )
        lbls.append( lbl )
        i += ( DIM + 1 )
            
  imgs = numpy.array( imgs )
  lbls = numpy.array( lbls )
  
  perm = numpy.arange( imgs.shape[ 0 ] )
  numpy.random.shuffle( perm )
  imgs = imgs[ perm ]
  lbls = lbls[ perm ]
  imgs = imgs.reshape( imgs.shape[ 0 ], imgs.shape[ 1 ], imgs.shape[ 2 ], 1 )

  print( "Loaded", imgs.shape[ 0 ], "images." )
  return imgs, lbls

def build_predictions( path, year, month, day ):
  tickers = load_tickers( path, YEAR_TO_GET_FROM )

  imgs = []
  lbls = []
  ticks = []
  lsts = []
  vols = []
  prcs = []
  
  for tc in tickers:
    lns = load_ticker( tc, path )
    i = find_date( lns, year, month, day )
    if i and i > ( DIM + 1 ):
      img, avg, lst = build_image( lns, i - DIM )
      lbl = build_label( lns, i, avg )
      imgs.append( img )
      lbls.append( lbl )
      ticks.append( tc[ : -1 ] )
      lsts.append( lst )
      vols.append( get_volume( lns[ i - 1 ] ) )
      prcs.append( get_adjusted_price( lns[ i - 1 ] ) )
  imgs = numpy.array( imgs )
  lbls = numpy.array( lbls )
  return imgs, lbls, ticks, lsts, prcs, vols
      
def build_today():
  tickers = load_tickers( STOCK_TODAY_DIR, now.year - 1)
  
  imgs = []
  ticks = []
  lsts = []

  for tc in tickers:
    lns = load_ticker( tc, STOCK_TODAY_DIR )
    i = find_date( lns, now.year, now.month, now.day )
    if i and i > ( DIM + 1 ):
      img, _, lst = build_image( lns, 1 + ( i - DIM ) )
      imgs.append( img )
      lsts.append( lst )
      ticks.append( tc[ : -1 ] )
  imgs = numpy.array( imgs )
  return imgs, ticks, lsts


class DataSet(object):

  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
      'images.shape: %s labels.shape: %s' % (images.shape,
                                             labels.shape))
    self._num_examples = images.shape[0]
    
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch( self, batch_size ):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets( path ):
  class DataSets(object):
    pass
  data_sets = DataSets()


  train_images, train_labels = build_chunks( path )
  TEST_SIZE = 1000



  
  test_images = train_images[:TEST_SIZE]
  test_labels = train_labels[:TEST_SIZE]
  train_images = train_images[TEST_SIZE:]
  train_labels = train_labels[TEST_SIZE:]
  
  
  data_sets.train = DataSet( train_images, train_labels )
  data_sets.test = DataSet( test_images, test_labels )

  return data_sets
  
         
